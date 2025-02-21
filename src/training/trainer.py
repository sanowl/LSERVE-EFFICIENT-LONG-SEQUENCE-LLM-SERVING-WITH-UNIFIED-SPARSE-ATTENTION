import torch
from typing import Dict, Optional
from ..serving.model_server import ModelServer
from ..data.data_processor import DataProcessor
from torch.utils.data import DataLoader
import logging

class ModelTrainer:
    def __init__(
        self,
        model: ModelServer,
        data_processor: DataProcessor,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda"
    ):
        self.model = model
        self.data_processor = data_processor
        self.optimizer = optimizer
        self.device = device
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        outputs = self.model.process_sequence(
            input_ids,
            sequence_id="train",
            attention_mask=attention_mask
        )
        
        loss = self.compute_loss(outputs, input_ids)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ):
        """Train for one epoch"""
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            loss = self.train_step(batch)
            total_loss += loss
            
            if batch_idx % 100 == 0:
                logging.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss:.4f}"
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @staticmethod
    def compute_loss(
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss"""
        return torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1)
        )
