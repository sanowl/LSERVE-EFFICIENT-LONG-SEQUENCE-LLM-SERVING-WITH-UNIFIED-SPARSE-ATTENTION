import torch
import torch.nn as nn
from typing import List, Tuple
import math

class AdaptiveChunking(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        min_chunk_size: int = 512,
        max_chunk_size: int = 2048,
        chunk_growth_factor: float = 1.5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_growth_factor = chunk_growth_factor
        
        # Chunk size predictor
        self.chunk_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def compute_chunk_boundaries(
        self,
        sequence: torch.Tensor,
        attention_density: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """Compute adaptive chunk boundaries based on attention density"""
        seq_length = sequence.size(1)
        
        # Predict chunk importance scores
        token_features = sequence.mean(dim=0)  # Average over batch
        chunk_scores = self.chunk_predictor(token_features)
        
        # Combine with attention density
        combined_scores = chunk_scores.squeeze() * attention_density
        
        # Initialize chunks
        chunks = []
        current_pos = 0
        current_size = self.min_chunk_size
        
        while current_pos < seq_length:
            # Adjust chunk size based on scores
            score_region = combined_scores[current_pos:current_pos + current_size].mean()
            adjusted_size = min(
                self.max_chunk_size,
                int(current_size * (1 + score_region * self.chunk_growth_factor))
            )
            
            end_pos = min(current_pos + adjusted_size, seq_length)
            chunks.append((current_pos, end_pos))
            
            current_pos = end_pos
            current_size = self.min_chunk_size
            
        return chunks
    
    def compute_attention_density(
        self,
        sequence: torch.Tensor,
        window_size: int = 128
    ) -> torch.Tensor:
        """Compute attention density estimation"""
        seq_length = sequence.size(1)
        density = torch.zeros(seq_length, device=sequence.device)
        
        # Compute local token relationships
        for i in range(0, seq_length, window_size):
            end_idx = min(i + window_size, seq_length)
            window = sequence[:, i:end_idx, :]
            
            # Compute local self-attention scores
            scores = torch.matmul(
                window, window.transpose(-1, -2)
            ) / math.sqrt(self.hidden_size)
            
            # Estimate density
            density[i:end_idx] = scores.mean(dim=(0, 1))
            
        return density
