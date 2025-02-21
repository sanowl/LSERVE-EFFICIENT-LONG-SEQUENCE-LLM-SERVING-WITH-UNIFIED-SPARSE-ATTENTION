from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    chunk_size: int = 1024
    overlap_size: int = 128
    max_memory_size: int = 1000
    cache_size: int = 10000
    window_size: int = 256
    global_tokens: int = 32
    sparsity_factor: float = 0.2
    attention_dropout: float = 0.1
    min_chunk_size: int = 512
    max_chunk_size: int = 2048
    chunk_growth_factor: float = 1.5
    prefetch_window: int = 512
    prediction_threshold: float = 0.7
