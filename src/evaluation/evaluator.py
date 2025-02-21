import torch
from typing import Dict, List, Tuple
from ..serving.model_server import ModelServer
import numpy as np
from torch.utils.data import DataLoader

class ModelEvaluator:
    def __init__(
        self,
        model: ModelServer,
        device: str = "cuda"
    ):
        self.model = model
        self.device = device
        
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        metrics = {
            "loss": 0.0,
            "perplexity": 0.0,
            "throughput": 0.0
        }
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            outputs = self.model.process_sequence(
                input_ids,
                sequence_id="eval",
                attention_mask=attention_mask
            )
            
            loss = self.compute_loss(outputs, input_ids)
            total_loss += loss.item()
            total_tokens += input_ids.numel()
        
        end_time.record()
        torch.cuda.synchronize()
        
        processing_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        
        metrics["loss"] = total_loss / len(dataloader)
        metrics["perplexity"] = torch.exp(torch.tensor(metrics["loss"])).item()
        metrics["throughput"] = total_tokens / processing_time
        
        return metrics
    
    @staticmethod
    def compute_loss(
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute evaluation loss"""
        return torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1)
        )
