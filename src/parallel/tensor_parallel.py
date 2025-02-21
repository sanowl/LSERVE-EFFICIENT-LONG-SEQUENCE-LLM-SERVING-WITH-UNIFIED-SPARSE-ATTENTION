import torch
import torch.distributed as dist
from typing import List, Optional, Tuple
import math

class TensorParallel:
    def __init__(
        self,
        model: torch.nn.Module,
        device_ids: List[int],
        output_device: Optional[int] = None
    ):
        self.model = model
        self.device_ids = device_ids
        self.num_gpus = len(device_ids)
        self.output_device = output_device or device_ids[0]
        
        # Initialize process groups
        self._init_process_groups()
        
        # Shard model across GPUs
        self._shard_parameters()
        
    def _init_process_groups(self):
        """Initialize process groups for tensor parallelism"""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        # Create process group for each tensor parallel dimension
        self.row_parallel_group = dist.new_group(self.device_ids)
        self.col_parallel_group = dist.new_group(self.device_ids)
    
    def _shard_parameters(self):
        """Shard model parameters across GPUs"""
        for name, param in self.model.named_parameters():
            if "query" in name or "key" in name or "value" in name:
                # Shard attention heads across GPUs
                shard_size = param.size(0) // self.num_gpus
                param.data = param.data.split(shard_size, dim=0)[dist.get_rank()]
            
            elif "ffn" in name:
                # Shard feed-forward layers
                if param.dim() > 1:
                    shard_size = param.size(0) // self.num_gpus
                    param.data = param.data.split(shard_size, dim=0)[dist.get_rank()]
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with tensor parallelism"""
        # Shard input across GPUs
        local_hidden_states = self._shard_tensor(hidden_states)
        
        # Local computation
        local_output = self.model(local_hidden_states)
        
        # Gather results
        output = self._gather_tensor(local_output)
        
        return output
    
    def _shard_tensor(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """Shard tensor across GPUs"""
        shard_size = tensor.size(-1) // self.num_gpus
        return tensor.split(shard_size, dim=-1)[dist.get_rank()]
    
    def _gather_tensor(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """Gather sharded tensor"""
        gathered = [torch.zeros_like(tensor) for _ in range(self.num_gpus)]
        dist.all_gather(gathered, tensor, group=self.col_parallel_group)
        return torch.cat(gathered, dim=-1)
    
    def cleanup(self):
        """Cleanup process groups"""
        dist.destroy_process_group(self.row_parallel_group)
        dist.destroy_process_group(self.col_parallel_group)
