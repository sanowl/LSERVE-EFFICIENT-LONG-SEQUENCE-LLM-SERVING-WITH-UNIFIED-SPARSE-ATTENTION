import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Optional, Tuple
import logging
from ..data.data_processor import DataProcessor
from ..utils.metrics_logger import MetricsLogger

class DistributedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        rank: int,
        world_size: int,
        local_rank: int,
        data_processor: DataProcessor,
        metrics_logger: MetricsLogger,
        device: str = "cuda"
    ):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"{device}:{local_rank}")
        
        # Initialize process group
        self._init_process_group()
        
        # Setup model for distributed training
        self.model = self._setup_model(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_processor = data_processor
        self.metrics_logger = metrics_logger
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
    def _init_process_group(self):
        """Initialize distributed process group"""
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank
        )
        
    def _setup_model(self, model: torch.nn.Module) -> DDP:
        """Setup model for distributed training"""
        # Move model to GPU
        model = model.to(self.device)
        
        # Enable sync batch norm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # Wrap model in DDP
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True
        )
        
        return model
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        grad_accum_steps: int
    ) -> float:
        """Single training step with gradient accumulation"""
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = self.model.forward(
                input_ids,
                attention_mask=attention_mask
            )
            loss = outputs.loss / grad_accum_steps
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        return loss.item() * grad_accum_steps
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping"""
        # Clip gradients
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )
        
        # Optimizer step with gradient scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        grad_accum_steps: int = 1
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_steps = 0
        
        # Reset dataloader sampler
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(dataloader):
            # Training step
            loss = self.train_step(batch, grad_accum_steps)
            total_loss += loss
            num_steps += 1
            
            # Optimizer step after gradient accumulation
            if (step + 1) % grad_accum_steps == 0:
                self.optimizer_step()
                
                # Log metrics (only on main process)
                if self.rank == 0 and num_steps % 10 == 0:
                    self.metrics_logger.log_metrics(
                        {
                            "train/loss": total_loss / num_steps,
                            "train/lr": self.scheduler.get_last_lr()[0]
                        },
                        step=epoch * len(dataloader) + step
                    )
        
        # Synchronize losses across GPUs
        avg_loss = torch.tensor(total_loss / num_steps, device=self.device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / self.world_size
        
        return {"loss": avg_loss}
    
    def save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        is_best: bool = False
    ):
        """Save distributed training checkpoint"""
        if self.rank == 0:  # Save only on main process
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict()
            }
            
            # Save checkpoint
            torch.save(checkpoint, save_path)
            
            # Save best model separately
            if is_best:
                best_path = os.path.join(
                    os.path.dirname(save_path),
                    "model_best.pth"
                )
                torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load distributed training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        # Load scaler state
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
        return checkpoint.get("epoch", 0)

    def cleanup(self):
        """Cleanup distributed training resources"""
        dist.destroy_process_group()