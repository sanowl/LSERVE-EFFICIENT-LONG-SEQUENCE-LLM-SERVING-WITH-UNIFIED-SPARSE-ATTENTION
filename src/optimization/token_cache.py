import torch
from typing import Dict, List, Optional, Tuple
import lru
import xxhash
from dataclasses import dataclass
import numpy as np
from threading import Lock

@dataclass
class CacheEntry:
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    timestamp: float
    frequency: int

class TokenCache:
    def __init__(
        self,
        cache_size: int = 100000,
        device: str = "cuda"
    ):
        self.device = device
        self.cache = lru.LRU(cache_size)
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
        
        # Frequency tracking for optimization
        self.token_frequencies: Dict[int, int] = {}
        self.sequence_patterns: Dict[str, int] = {}
    
    def cache_lookup(
        self,
        text: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Lookup text in cache"""
        with self.lock:
            cache_key = self._compute_hash(text)
            entry = self.cache.get(cache_key)
            
            if entry is not None:
                self.hits += 1
                entry.frequency += 1
                return entry.tokens, entry.attention_mask
            
            self.misses += 1
            return None
    
    def cache_store(
        self,
        text: str,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """Store tokenization results in cache"""
        with self.lock:
            cache_key = self._compute_hash(text)
            entry = CacheEntry(
                tokens=tokens,
                attention_mask=attention_mask,
                timestamp=time.time(),
                frequency=1
            )
            self.cache[cache_key] = entry
            
            # Update token frequencies
            for token in tokens.unique():
                self.token_frequencies[token.item()] = \
                    self.token_frequencies.get(token.item(), 0) + 1
    
    def _compute_hash(self, text: str) -> str:
        """Compute hash for cache key"""
        return xxhash.xxh64(text.encode()).hexdigest()
    
    def optimize_cache(self):
        """Optimize cache based on usage patterns"""
        with self.lock:
            # Remove infrequently used entries
            threshold = np.percentile(
                [entry.frequency for entry in self.cache.values()],
                25
            )
            
            to_remove = []
            for key, entry in self.cache.items():
                if entry.frequency < threshold:
                    to_remove.append(key)
            
            for key in to_remove:
                del self.cache[key]
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache performance statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "hit_rate": hit_rate,
                "cache_size": len(self.cache),
                "unique_tokens": len(self.token_frequencies),
                "memory_usage_mb": self._estimate_memory_usage() / 1024**2
            }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate cache memory usage in bytes"""
        total_bytes = 0
        for entry in self.cache.values():
            total_bytes += entry.tokens.element_size() * entry.tokens.nelement()
            total_bytes += entry.attention_mask.element_size() * entry.attention_mask.nelement()
        return total_bytes
