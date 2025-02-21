import torch
from typing import Dict, List, Optional
import numpy as np

class MemoryManager:
    def __init__(self, max_memory_size: int, hidden_size: int):
        self.max_memory_size = max_memory_size
        self.hidden_size = hidden_size
        self.memory_bank = {}
        self.access_history = {}
        
    def store_memory(self, key: str, tensor: torch.Tensor):
        """Store tensor in memory bank with LRU tracking"""
        if len(self.memory_bank) >= self.max_memory_size:
            self._evict_least_used()
            
        self.memory_bank[key] = tensor
        self.access_history[key] = 0
        
    def retrieve_memory(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve tensor from memory bank and update access count"""
        if key in self.memory_bank:
            self.access_history[key] += 1
            return self.memory_bank[key]
        return None
    
    def _evict_least_used(self):
        """Evict least recently used items from memory"""
        if not self.memory_bank:
            return
            
        min_access = min(self.access_history.values())
        keys_to_evict = [
            k for k, v in self.access_history.items() 
            if v == min_access
        ]
        
        # Evict least accessed item
        key_to_evict = keys_to_evict[0]
        del self.memory_bank[key_to_evict]
        del self.access_history[key_to_evict]
        
    def clear_memory(self):
        """Clear all stored memories"""
        self.memory_bank.clear()
        self.access_history.clear()
