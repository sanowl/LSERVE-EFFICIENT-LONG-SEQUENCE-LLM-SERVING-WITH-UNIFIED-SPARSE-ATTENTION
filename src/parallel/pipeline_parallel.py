import torch
from typing import List, Dict, Optional
from torch.nn import Module
from queue import Queue
import threading

class PipelineParallel:
    def __init__(
        self,
        num_stages: int,
        model_chunks: List[Module],
        batch_size: int,
        micro_batch_size: int,
        device_map: Dict[int, str]
    ):
        self.num_stages = num_stages
        self.model_chunks = model_chunks
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.device_map = device_map
        
        # Initialize queues for pipeline stages
        self.queues = [Queue() for _ in range(num_stages + 1)]
        self.grad_queues = [Queue() for _ in range(num_stages + 1)]
        
        # Move model chunks to respective devices
        for i, chunk in enumerate(model_chunks):
            chunk.to(self.device_map[i])
    
    def forward_stage(self, stage_id: int):
        """Execute forward pass for a pipeline stage"""
        model_chunk = self.model_chunks[stage_id]
        input_queue = self.queues[stage_id]
        output_queue = self.queues[stage_id + 1]
        
        while True:
            micro_batch = input_queue.get()
            if micro_batch is None:  # Exit signal
                output_queue.put(None)
                break
                
            with torch.cuda.device(self.device_map[stage_id]):
                output = model_chunk(micro_batch)
                output_queue.put(output)
    
    def backward_stage(self, stage_id: int):
        """Execute backward pass for a pipeline stage"""
        model_chunk = self.model_chunks[stage_id]
        grad_queue = self.grad_queues[stage_id]
        prev_grad_queue = self.grad_queues[stage_id - 1]
        
        while True:
            grads = grad_queue.get()
            if grads is None:  # Exit signal
                prev_grad_queue.put(None)
                break
                
            with torch.cuda.device(self.device_map[stage_id]):
                output = model_chunk.backward(grads)
                prev_grad_queue.put(output)
    
    def run_pipeline(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Execute full pipeline parallel forward and backward pass"""
        num_micro_batches = self.batch_size // self.micro_batch_size
        
        # Start forward threads
        forward_threads = []
        for stage_id in range(self.num_stages):
            thread = threading.Thread(
                target=self.forward_stage,
                args=(stage_id,)
            )
            thread.start()
            forward_threads.append(thread)
        
        # Feed input micro-batches
        for i in range(num_micro_batches):
            start_idx = i * self.micro_batch_size
            end_idx = start_idx + self.micro_batch_size
            micro_batch = input_batch[start_idx:end_idx]
            self.queues[0].put(micro_batch)
        
        # Signal completion
        self.queues[0].put(None)
        
        # Wait for forward completion
        for thread in forward_threads:
            thread.join()
        
        # Start backward threads
        backward_threads = []
        for stage_id in reversed(range(self.num_stages)):
            thread = threading.Thread(
                target=self.backward_stage,
                args=(stage_id,)
            )
            thread.start()
            backward_threads.append(thread)
        
        # Wait for backward completion
        for thread in backward_threads:
            thread.join()
        
        # Collect outputs
        outputs = []
        while not self.queues[-1].empty():
            output = self.queues[-1].get()
            if output is not None:
                outputs.append(output)
        
        return torch.cat(outputs, dim=0)
