import torch
import asyncio
from typing import AsyncGenerator, Dict, Optional
from dataclasses import dataclass
from queue import Queue
import numpy as np

@dataclass
class StreamingConfig:
    chunk_size: int = 512
    stride: int = 256
    max_tokens_per_batch: int = 2048
    prefetch_size: int = 2

class StreamProcessor:
    def __init__(
        self,
        model: torch.nn.Module,
        config: StreamingConfig
    ):
        self.model = model
        self.config = config
        self.prefetch_queue = Queue(maxsize=config.prefetch_size)
        
    async def process_stream(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> AsyncGenerator[torch.Tensor, None]:
        """Process long sequence in streaming fashion"""
        sequence_length = input_ids.size(1)
        
        # Initialize KV-cache
        kv_cache = self._init_kv_cache()
        
        for chunk_start in range(0, sequence_length, self.config.stride):
            chunk_end = min(chunk_start + self.config.chunk_size, sequence_length)
            
            # Get current chunk
            chunk_input = input_ids[:, chunk_start:chunk_end]
            chunk_mask = attention_mask[:, chunk_start:chunk_end]
            
            # Process chunk with cached attention
            with torch.cuda.amp.autocast():
                output = await self._process_chunk(
                    chunk_input,
                    chunk_mask,
                    kv_cache,
                    chunk_start
                )
            
            # Update KV-cache
            self._update_kv_cache(kv_cache, output, chunk_start)
            
            # Yield results
            yield output
            
            # Prefetch next chunk if available
            if chunk_end < sequence_length:
                next_start = chunk_end
                next_end = min(next_start + self.config.chunk_size, sequence_length)
                asyncio.create_task(self._prefetch_chunk(
                    input_ids[:, next_start:next_end],
                    attention_mask[:, next_start:next_end]
                ))
    
    async def _process_chunk(
        self,
        chunk_input: torch.Tensor,
        chunk_mask: torch.Tensor,
        kv_cache: Dict[str, torch.Tensor],
        chunk_start: int
    ) -> torch.Tensor:
        """Process single chunk with caching"""
        # Check prefetch queue first
        if not self.prefetch_queue.empty():
            return await self.prefetch_queue.get()
        
        # Process chunk with attention
        return self.model(
            chunk_input,
            attention_mask=chunk_mask,
            kv_cache=kv_cache,
            cache_offset=chunk_start
        )
    
    async def _prefetch_chunk(
        self,
        chunk_input: torch.Tensor,
        chunk_mask: torch.Tensor
    ):
        """Prefetch and cache next chunk"""
        with torch.no_grad():
            output = self.model(
                chunk_input,
                attention_mask=chunk_mask
            )
        await self.prefetch_queue.put(output)
    
    def _init_kv_cache(self) -> Dict[str, torch.Tensor]:
        """Initialize key-value cache"""
        return {
            "keys": torch.zeros(
                (self.config.max_tokens_per_batch, self.model.config.hidden_size),
                device="cuda"
            ),
            "values": torch.zeros(
                (self.config.max_tokens_per_batch, self.model.config.hidden_size),
                device="cuda"
            )
        }
    
    def _update_kv_cache(
        self,
        kv_cache: Dict[str, torch.Tensor],
        output: torch.Tensor,
        start_idx: int
    ):
        """Update key-value cache with new computed values"""
        cache_size = kv_cache["keys"].size(0)
        update_size = min(output.size(1), cache_size - start_idx)
        
        kv_cache["keys"][start_idx:start_idx + update_size] = \
            output[:, :update_size, :]
        kv_cache["values"][start_idx:start_idx + update_size] = \
            output[:, :update_size, :]
