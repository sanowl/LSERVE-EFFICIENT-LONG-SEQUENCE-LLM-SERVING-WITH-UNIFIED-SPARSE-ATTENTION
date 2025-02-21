import torch
from typing import Dict, List, Optional
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

class ModelSharding:
    def __init__(
        self,
        model: torch.nn.Module,
        num_shards: int,
        overlap_layers: int = 1
    ):
        self.model = model
        self.num_shards = num_shards
        self.overlap_layers = overlap_layers
        
        # Initialize shard mappings
        self.layer_to_shard = {}
        self.shard_to_layers = {i: [] for i in range(num_shards)}
        
        self._create_shards()
    
    def _create_shards(self):
        """Distribute model layers across shards"""
        layers = list(self.model.named_modules())
        layers_per_shard = len(layers) // self.num_shards
        
        for i, (name, layer) in enumerate(layers):
            shard_id = min(i // layers_per_shard, self.num_shards - 1)
            self.layer_to_shard[name] = shard_id
            self.shard_to_layers[shard_id].append((name, layer))
            
            # Add overlapping layers
            if i % layers_per_shard < self.overlap_layers and shard_id > 0:
                self.shard_to_layers[shard_id - 1].append((name, layer))
    
    def get_shard(self, shard_id: int) -> torch.nn.Module:
        """Get model shard for given shard ID"""
        shard_layers = self.shard_to_layers[shard_id]
        shard = torch.nn.ModuleList([layer for _, layer in shard_layers])
        return shard
    
    def optimize_memory(self):
        """Apply memory optimizations to shards"""
        for shard_id in range(self.num_shards):
            shard = self.get_shard(shard_id)
            
            # Enable gradient checkpointing
            shard.gradient_checkpointing_enable()
            
            # Apply mixed precision
            shard = torch.cuda.amp.autocast()(shard)
            
            # Optimize memory allocation
            torch.cuda.empty_cache()
    
    def load_balance_shards(self):
        """Balance computation across shards"""
        total_params = sum(p.numel() for p in self.model.parameters())
        target_params_per_shard = total_params // self.num_shards
        
        current_shard = 0
        current_params = 0
        
        new_shard_mapping = {}
        
        for name, layer in self.model.named_modules():
            layer_params = sum(p.numel() for p in layer.parameters())
            
            if current_params + layer_params > target_params_per_shard:
                current_shard += 1
                current_params = 0
            
            new_shard_mapping[name] = min(current_shard, self.num_shards - 1)
            current_params += layer_params
        
        self.layer_to_shard = new_shard_mapping
        self._update_shard_layers()
    
    def _update_shard_layers(self):
        """Update shard layer assignments"""
        self.shard_to_layers = {i: [] for i in range(self.num_shards)}
        
        for name, layer in self.model.named_modules():
            shard_id = self.layer_to_shard[name]
            self.shard_to_layers[shard_id].append((name, layer))
