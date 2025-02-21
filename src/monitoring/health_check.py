import torch
import asyncio
from typing import Dict, Optional
import time
import logging
from dataclasses import dataclass

@dataclass
class HealthStatus:
    is_healthy: bool
    gpu_available: bool
    memory_ok: bool
    latency_ok: bool
    error_message: Optional[str] = None

class HealthChecker:
    def __init__(
        self,
        max_latency_ms: float = 1000.0,
        max_gpu_memory_percent: float = 95.0,
        check_interval: float = 5.0
    ):
        self.max_latency_ms = max_latency_ms
        self.max_gpu_memory_percent = max_gpu_memory_percent
        self.check_interval = check_interval
        self.is_running = False
        
    async def start_monitoring(self):
        """Start health monitoring loop"""
        self.is_running = True
        while self.is_running:
            status = await self.check_health()
            if not status.is_healthy:
                logging.warning(f"Health check failed: {status.error_message}")
            await asyncio.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_running = False
    
    async def check_health(self) -> HealthStatus:
        """Perform health check"""
        try:
            # Check GPU availability
            if not torch.cuda.is_available():
                return HealthStatus(
                    is_healthy=False,
                    gpu_available=False,
                    memory_ok=False,
                    latency_ok=False,
                    error_message="GPU not available"
                )
            
            # Check GPU memory
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            memory_ok = memory_used * 100 < self.max_gpu_memory_percent
            
            # Check model latency
            latency = await self._measure_inference_latency()
            latency_ok = latency < self.max_latency_ms
            
            is_healthy = memory_ok and latency_ok
            error_message = None
            if not is_healthy:
                error_message = f"Memory used: {memory_used:.1f}%, Latency: {latency:.1f}ms"
            
            return HealthStatus(
                is_healthy=is_healthy,
                gpu_available=True,
                memory_ok=memory_ok,
                latency_ok=latency_ok,
                error_message=error_message
            )
            
        except Exception as e:
            return HealthStatus(
                is_healthy=False,
                gpu_available=True,
                memory_ok=False,
                latency_ok=False,
                error_message=str(e)
            )
    
    async def _measure_inference_latency(self) -> float:
        """Measure model inference latency"""
        # Create small test input
        test_input = torch.randn(1, 512, device="cuda")
        
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = self.model(test_input)
        end_time = time.perf_counter()
        
        return (end_time - start_time) * 1000  # Convert to ms
