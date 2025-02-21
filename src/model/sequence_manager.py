import torch
from typing import List, Tuple

class SequenceManager:
    def __init__(self, chunk_size: int, overlap_size: int):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
    
    def chunk_sequence(self, sequence: torch.Tensor) -> List[torch.Tensor]:
        """Split long sequence into overlapping chunks"""
        seq_length = sequence.size(1)
        chunks = []
        
        for start in range(0, seq_length, self.chunk_size - self.overlap_size):
            end = min(start + self.chunk_size, seq_length)
            chunk = sequence[:, start:end, :]
            chunks.append(chunk)
            
            if end == seq_length:
                break
                
        return chunks
    
    def merge_chunks(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        """Merge processed chunks back into a single sequence"""
        batch_size = chunks[0].size(0)
        hidden_size = chunks[0].size(-1)
        
        # Calculate total sequence length
        total_length = (len(chunks) - 1) * (self.chunk_size - self.overlap_size)
        total_length += chunks[-1].size(1)
        
        # Initialize output tensor
        merged = torch.zeros(
            (batch_size, total_length, hidden_size),
            device=chunks[0].device,
            dtype=chunks[0].dtype
        )
        
        # Merge chunks with overlap handling
        position = 0
        for i, chunk in enumerate(chunks):
            if i == 0:
                merged[:, :chunk.size(1), :] = chunk
                position = chunk.size(1) - self.overlap_size
            else:
                # Blend overlapping regions
                overlap_start = position
                overlap_end = position + self.overlap_size
                chunk_end = position + chunk.size(1) - self.overlap_size
                
                # Linear interpolation in overlapping region
                alpha = torch.linspace(0, 1, self.overlap_size, device=chunk.device)
                alpha = alpha.view(1, -1, 1)
                
                merged[:, overlap_start:overlap_end, :] = (
                    (1 - alpha) * merged[:, overlap_start:overlap_end, :] +
                    alpha * chunk[:, :self.overlap_size, :]
                )
                
                # Copy non-overlapping region
                merged[:, overlap_end:chunk_end, :] = chunk[:, self.overlap_size:, :]
                
                position = chunk_end
                
        return merged
