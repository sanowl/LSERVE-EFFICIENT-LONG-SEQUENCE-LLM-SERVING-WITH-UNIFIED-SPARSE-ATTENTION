import torch
from typing import Dict, Optional, List
from dataclasses import dataclass
import numpy as np

@dataclass
class GraphConfig:
    batch_sizes: List[int]
    sequence_lengths: List[int]
    warmup_iterations: int = 5

class CUDAGraphOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: GraphConfig
    ):
        self.model = model
        self.config = config
        self.graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        
        # Initialize graphs for common sizes
        self._init_graphs()
    
    def _init_graphs(self):
        """Initialize CUDA graphs for common input sizes"""
        for batch_size in self.config.batch_sizes:
            for seq_length in self.config.sequence_lengths:
                self._create_graph(batch_size, seq_length)
    
    def _create_graph(
        self,
        batch_size: int,
        seq_length: int
    ):
        """Create CUDA graph for specific input size"""
        # Create static inputs
        static_inputs = {
            "input_ids": torch.zeros(
                (batch_size, seq_length),
                dtype=torch.long,
                device="cuda"
            ),
            "attention_mask": torch.ones(
                (batch_size, seq_length),
                dtype=torch.float,
                device="cuda"
            )
        }
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            _ = self.model(**static_inputs)
        
        # Create graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_outputs = self.model(**static_inputs)
        
        # Store graph and static tensors
        key = (batch_size, seq_length)
        self.graphs[key] = graph
        self.static_inputs[key] = static_inputs
        
    def optimize_inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Run inference with CUDA graph if available"""
        batch_size, seq_length = input_ids.shape
        key = (batch_size, seq_length)
        
        if key in self.graphs:
            # Copy inputs to static tensors
            self.static_inputs[key]["input_ids"].copy_(input_ids)
            self.static_inputs[key]["attention_mask"].copy_(attention_mask)
            
            # Replay graph
            self.graphs[key].replay()
            
            return self.static_inputs[key]["output"]
        
        return None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage of stored graphs"""
        total_memory = 0
        for key, inputs in self.static_inputs.items():
            for tensor in inputs.values():
                total_memory += tensor.element_size() * tensor.nelement()
        
        return {
            "num_graphs": len(self.graphs),
            "total_memory_mb": total_memory / (1024 * 1024)
        }
