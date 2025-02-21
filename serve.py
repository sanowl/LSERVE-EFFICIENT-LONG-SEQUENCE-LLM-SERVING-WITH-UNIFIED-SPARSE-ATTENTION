import torch
import argparse
import yaml
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from src.serving.model_server import ModelServer
from src.config.config_validation import validate_config_file
from src.serving.dynamic_batcher import DynamicBatcher
from src.utils.model_utils import initialize_model, load_model_checkpoint
import uvicorn
import asyncio
import logging

class InferenceRequest(BaseModel):
    text: str
    max_length: Optional[int] = 2048
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.9
    sequence_id: Optional[str] = None

class InferenceResponse(BaseModel):
    text: str
    latency_ms: float

async def setup_model(config_path: str, checkpoint_path: str):
    # Load configuration
    config = validate_config_file(config_path)
    
    # Initialize model
    model = initialize_model(config["attention"])
    model = load_model_checkpoint(model, checkpoint_path)
    model.eval()
    
    # Initialize dynamic batcher
    batcher = DynamicBatcher(
        max_batch_size=config["serving"]["max_batch_size"],
        max_wait_time=config["serving"]["timeout_ms"] / 1000,
        min_batch_size=1
    )
    
    # Initialize model server
    server = ModelServer(
        model=model,
        batcher=batcher,
        config=config
    )
    
    return server

app = FastAPI()
model_server = None

@app.on_event("startup")
async def startup():
    global model_server
    model_server = await setup_model(
        "config/runtime_config.yaml",
        "checkpoints/model_best.pt"
    )

@app.post("/v1/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest):
    global model_server
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    output = await model_server.generate_async(
        text=request.text,
        max_length=request.max_length,
        temperature=request.temperature,
        top_p=request.top_p,
        sequence_id=request.sequence_id
    )
    end_time.record()
    
    torch.cuda.synchronize()
    latency_ms = start_time.elapsed_time(end_time)
    
    return InferenceResponse(
        text=output,
        latency_ms=latency_ms
    )

@app.get("/v1/health")
async def health_check():
    return {"status": "healthy"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    uvicorn.run(
        "serve:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )

if __name__ == "__main__":
    main()
