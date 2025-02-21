import torch
from typing import List, Dict, Tuple, Optional
from transformers import PreTrainedTokenizer

class DataProcessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_sequence_length: int = 16384
    ):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        
    def preprocess_text(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Preprocess raw text into model inputs"""
        encoded = self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_sequence_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
    
    def create_sliding_windows(
        self,
        input_ids: torch.Tensor,
        window_size: int,
        stride: int
    ) -> List[torch.Tensor]:
        """Create sliding windows for long sequence processing"""
        seq_length = input_ids.size(1)
        windows = []
        
        for start in range(0, seq_length - window_size + 1, stride):
            end = start + window_size
            window = input_ids[:, start:end]
            windows.append(window)
            
        return windows
