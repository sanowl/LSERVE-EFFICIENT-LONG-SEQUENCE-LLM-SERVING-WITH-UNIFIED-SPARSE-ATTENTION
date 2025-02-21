import torch
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from collections import deque
import psutil
import GPUtil
import logging

@dataclass
class ResourceMetrics:
    gpu_utilization: float
    gpu_memory_used: float
    cpu_utilization: float
    system_memory_used: float
    cuda_memory_allocated: float
    cuda_memory_cached: float

class PerformanceTracker:
    def __init__(
        self,
        window_size: int = 1000,
        log_frequency: int = 100
    ):
        self.window_size = window_size
        self.log_frequency = log_frequency
        
        # Metrics storage
        self.latencies = deque(maxlen=window_size)
        self.throughputs = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        
        # Performance counters
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()
        
    def update_metrics(
        self,
        batch_size: int,
        sequence_length: int,
        latency: float
    ):
        """Update performance metrics"""
        self.latencies.append(latency)
        self.batch_sizes.append(batch_size)
        
        tokens_per_second = (batch_size * sequence_length) / (latency / 1000)
        self.throughputs.append(tokens_per_second)
        
        self.total_requests += batch_size
        self.total_tokens += batch_size * sequence_length
        
        # Log metrics periodically
        if self.total_requests % self.log_frequency == 0:
            self.log_current_metrics()
    
    def get_resource_metrics(self) -> ResourceMetrics:
        """Get current system resource utilization"""
        # GPU metrics
        gpu = GPUtil.getGPUs()[0]  # Assuming first GPU
        gpu_util = gpu.load * 100
        gpu_mem = gpu.memoryUtil * 100
        
        # CPU metrics
        cpu_util = psutil.cpu_percent()
        sys_mem = psutil.virtual_memory().percent
        
        # CUDA memory
        cuda_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        cuda_cached = torch.cuda.memory_reserved() / 1024**2  # MB
        
        return ResourceMetrics(
            gpu_utilization=gpu_util,
            gpu_memory_used=gpu_mem,
            cpu_utilization=cpu_util,
            system_memory_used=sys_mem,
            cuda_memory_allocated=cuda_allocated,
            cuda_memory_cached=cuda_cached
        )
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        if not self.latencies:
            return {}
        
        metrics = {
            "mean_latency_ms": np.mean(self.latencies),
            "p90_latency_ms": np.percentile(self.latencies, 90),
            "p99_latency_ms": np.percentile(self.latencies, 99),
            "mean_throughput": np.mean(self.throughputs),
            "mean_batch_size": np.mean(self.batch_sizes),
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "uptime_hours": (time.time() - self.start_time) / 3600
        }
        
        # Add resource metrics
        resources = self.get_resource_metrics()
        metrics.update({
            "gpu_utilization": resources.gpu_utilization,
            "gpu_memory_used": resources.gpu_memory_used,
            "cpu_utilization": resources.cpu_utilization,
            "system_memory_used": resources.system_memory_used,
            "cuda_memory_allocated_mb": resources.cuda_memory_allocated,
            "cuda_memory_cached_mb": resources.cuda_memory_cached
        })
        
        return metrics
    
    def log_current_metrics(self):
        """Log current performance metrics"""
        metrics = self.get_performance_metrics()
        logging.info("Performance Metrics:")
        for name, value in metrics.items():
            logging.info(f"  {name}: {value:.2f}")
