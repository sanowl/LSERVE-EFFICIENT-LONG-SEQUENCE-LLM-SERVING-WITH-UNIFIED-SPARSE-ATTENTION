import torch
import time
from typing import Dict, List, Optional
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from ..model.unified_sparse_attention import UnifiedSparseAttention

class ModelProfiler:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.timing_stats = {}
        self.memory_stats = {}
        self.flop_stats = {}
        
    def profile_memory(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Profile memory usage"""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Warmup
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        torch.cuda.reset_peak_memory_stats()
        
        # Profile run
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        max_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        return {
            "peak_memory_mb": max_memory / (1024 * 1024),
            "current_memory_mb": current_memory / (1024 * 1024)
        }
    
    def profile_attention_patterns(self, model: UnifiedSparseAttention) -> Dict[str, float]:
        """Profile attention pattern statistics"""
        stats = {}
        
        # Analyze sparsity patterns
        with torch.no_grad():
            attention_scores = model.last_attention_scores
            if attention_scores is not None:
                sparsity = (attention_scores == 0).float().mean().item()
                avg_attention = attention_scores.mean().item()
                max_attention = attention_scores.max().item()
                
                stats.update({
                    "attention_sparsity": sparsity,
                    "avg_attention_score": avg_attention,
                    "max_attention_score": max_attention
                })
        
        return stats
    
    def profile_throughput(
        self,
        input_tensor: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Profile model throughput"""
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(input_tensor)
        
        # Profile runs
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        latencies = []
        
        for _ in range(num_runs):
            start_event.record()
            with torch.no_grad():
                _ = self.model(input_tensor)
            end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        
        latencies = np.array(latencies)
        
        return {
            "avg_latency_ms": np.mean(latencies),
            "p90_latency_ms": np.percentile(latencies, 90),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput_seq_per_sec": 1000 / np.mean(latencies)
        }
    
    def profile_detailed(self, input_tensor: torch.Tensor) -> str:
        """Generate detailed profiling report"""
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                _ = self.model(input_tensor)
        
        # Generate report
        report = prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20
        )
        
        return report
    
    def export_trace(self, input_tensor: torch.Tensor, path: str):
        """Export trace for visualization"""
        with torch.profiler.profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_trace=True
        ) as prof:
            _ = self.model(input_tensor)
            
        prof.export_chrome_trace(path)
