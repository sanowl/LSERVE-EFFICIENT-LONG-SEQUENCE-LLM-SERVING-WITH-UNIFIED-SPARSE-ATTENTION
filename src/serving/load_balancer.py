import asyncio
from typing import List, Dict, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch

class LoadBalancer:
    def __init__(
        self,
        num_gpus: int,
        max_sequence_length: int = 16384,
        target_gpu_util: float = 0.8
    ):
        self.num_gpus = num_gpus
        self.max_sequence_length = max_sequence_length
        self.target_gpu_util = target_gpu_util
        
        # Track GPU utilization
        self.gpu_utilization = np.zeros(num_gpus)
        self.sequence_counts = np.zeros(num_gpus)
        self.gpu_locks = [asyncio.Lock() for _ in range(num_gpus)]
        
        # Initialize thread pool for non-blocking operations
        self.executor = ThreadPoolExecutor(max_workers=num_gpus)
    
    async def get_optimal_gpu(self, sequence_length: int) -> int:
        """Select best GPU for given sequence length"""
        scores = []
        for gpu_id in range(self.num_gpus):
            # Calculate GPU score based on:
            # 1. Current utilization
            # 2. Number of active sequences
            # 3. Available memory
            util_score = 1.0 - (self.gpu_utilization[gpu_id] / self.target_gpu_util)
            seq_score = 1.0 - (self.sequence_counts[gpu_id] / 100)  # Arbitrary limit
            mem_score = self.get_gpu_memory_score(gpu_id)
            
            score = (util_score * 0.4 + seq_score * 0.3 + mem_score * 0.3)
            scores.append(score)
        
        return int(np.argmax(scores))
    
    def get_gpu_memory_score(self, gpu_id: int) -> float:
        """Get memory availability score for GPU"""
        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
        used_mem = torch.cuda.memory_allocated(gpu_id)
        return 1.0 - (used_mem / total_mem)
    
    async def update_gpu_stats(self, gpu_id: int, sequence_length: int):
        """Update GPU utilization statistics"""
        async with self.gpu_locks[gpu_id]:
            self.sequence_counts[gpu_id] += 1
            self.gpu_utilization[gpu_id] = await self.measure_gpu_utilization(gpu_id)
    
    async def measure_gpu_utilization(self, gpu_id: int) -> float:
        """Measure current GPU utilization"""
        loop = asyncio.get_event_loop()
        # Run in thread pool to avoid blocking
        return await loop.run_in_executor(
            self.executor,
            lambda: torch.cuda.utilization(gpu_id)
        )
    
    async def release_gpu(self, gpu_id: int):
        """Release GPU resources after processing"""
        async with self.gpu_locks[gpu_id]:
            self.sequence_counts[gpu_id] -= 1
            self.gpu_utilization[gpu_id] = await self.measure_gpu_utilization(gpu_id)
