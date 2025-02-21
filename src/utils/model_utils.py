import torch
from typing import Optional, Dict
from ..serving.model_server import ModelServer
from ...config.model_config import ModelConfig

def initialize_model(config: ModelConfig) -> ModelServer:
    """Initialize model server with configuration"""
    model = ModelServer(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        chunk_size=config.chunk_size,
        overlap_size=config.overlap_size,
        max_memory_size=config.max_memory_size,
        cache_size=config.cache_size
    )
    return model

def save_model_checkpoint(
    model: ModelServer,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None
):
    """Save model checkpoint"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None
    }
    torch.save(checkpoint, save_path)

def load_model_checkpoint(
    model: ModelServer,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> ModelServer:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return model
