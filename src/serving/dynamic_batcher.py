import torch
from typing import List, Dict, Optional
import time
from queue import PriorityQueue
import threading

class DynamicBatcher:
    def __init__(
        self,
        max_batch_size: int,
        max_wait_time: float,
        min_batch_size: int = 1
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size
        
        self.request_queue = PriorityQueue()
        self.current_batch = []
        self.batch_ready = threading.Event()
        
        self._start_batching_thread()
    
    def _start_batching_thread(self):
        """Start background thread for batch formation"""
        self.batching_thread = threading.Thread(
            target=self._batch_formation_loop
        )
        self.batching_thread.daemon = True
        self.batching_thread.start()
    
    def _batch_formation_loop(self):
        """Main loop for dynamic batch formation"""
        while True:
            batch_start_time = time.time()
            current_batch_size = 0
            
            while current_batch_size < self.max_batch_size:
                # Check if we should create a batch
                time_elapsed = time.time() - batch_start_time
                if (current_batch_size >= self.min_batch_size and 
                    time_elapsed >= self.max_wait_time):
                    break
                
                try:
                    # Get next request with timeout
                    priority, request = self.request_queue.get(
                        timeout=self.max_wait_time
                    )
                    self.current_batch.append(request)
                    current_batch_size += 1
                except:
                    if current_batch_size >= self.min_batch_size:
                        break
            
            if current_batch_size > 0:
                self.batch_ready.set()
                self.batch_ready.clear()
    
    def add_request(
        self,
        request: Dict[str, torch.Tensor],
        priority: float = 0.0
    ):
        """Add request to batching queue"""
        self.request_queue.put((priority, request))
    
    def get_next_batch(self) -> Optional[List[Dict[str, torch.Tensor]]]:
        """Get next formed batch"""
        if not self.current_batch:
            self.batch_ready.wait(timeout=self.max_wait_time)
        
        if self.current_batch:
            batch = self.current_batch
            self.current_batch = []
            return batch
        
        return None
    
    def optimize_batch_size(self, throughput_history: List[float]):
        """Dynamically adjust batch size based on throughput"""
        if len(throughput_history) < 2:
            return
            
        current_throughput = throughput_history[-1]
        previous_throughput = throughput_history[-2]
        
        # Adjust batch size based on throughput changes
        if current_throughput > previous_throughput * 1.1:
            self.max_batch_size = min(
                self.max_batch_size * 2,
                self.max_batch_size
            )
        elif current_throughput < previous_throughput * 0.9:
            self.max_batch_size = max(
                self.max_batch_size // 2,
                self.min_batch_size
            )
