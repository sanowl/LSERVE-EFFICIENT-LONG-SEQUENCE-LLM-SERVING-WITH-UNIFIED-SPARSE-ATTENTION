import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from lserve_cpp import attention_kernels

class UnifiedSparseAttention(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 num_attention_heads: int,
                 window_size: int = 256,
                 global_tokens: int = 32,
                 sparsity_factor: float = 0.2,
                 attention_dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(attention_dropout)
        
        self.window_size = window_size
        self.global_tokens = global_tokens
        self.sparsity_factor = sparsity_factor
        
        # Add projections for global tokens
        self.global_query = nn.Linear(hidden_size, self.all_head_size)
        self.global_key = nn.Linear(hidden_size, self.all_head_size)
        self.global_value = nn.Linear(hidden_size, self.all_head_size)
        
        # Add locality sensitive hashing
        self.lsh_num_buckets = 32
        self.lsh_hash_size = 16
        self.lsh_projection = nn.Parameter(
            torch.randn(hidden_size, self.lsh_hash_size)
        )
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def compute_lsh_patterns(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute LSH-based attention patterns"""
        # Project hidden states for hashing
        hash_projections = torch.matmul(hidden_states, self.lsh_projection)
        hash_codes = torch.sign(hash_projections)  # Binary codes
        
        # Compute bucket assignments
        bucket_ids = torch.sum(
            2 ** torch.arange(
                self.lsh_hash_size, 
                device=hash_codes.device
            )[None, :] * (hash_codes > 0),
            dim=-1
        )
        bucket_ids = bucket_ids % self.lsh_num_buckets
        
        return bucket_ids
    
    def compute_sliding_window_mask(
        self,
        seq_length: int,
        window_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Compute sliding window attention mask"""
        positions = torch.arange(seq_length, device=device)
        distances = positions[None, :] - positions[:, None]
        window_mask = (distances.abs() <= window_size//2)
        
        return window_mask.float()
    
    def compute_sparse_attention(self, query_layer, key_layer, value_layer, attention_mask=None, hidden_states=None):
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Sparse attention implementation
        sparse_mask = self.generate_sparse_mask(attention_scores, hidden_states)
        attention_scores = attention_scores * sparse_mask
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer
    
    def generate_sparse_mask(
        self,
        attention_scores: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Generate sophisticated sparse attention mask"""
        batch_size, num_heads, seq_length, _ = attention_scores.size()
        device = attention_scores.device
        
        # 1. Sliding window mask
        window_mask = self.compute_sliding_window_mask(
            seq_length, self.window_size, device
        )
        
        # 2. LSH-based patterns
        bucket_ids = self.compute_lsh_patterns(hidden_states)
        lsh_mask = (
            bucket_ids.unsqueeze(1) == bucket_ids.unsqueeze(2)
        ).float()
        
        # 3. Global tokens mask (allow attention to/from global tokens)
        global_mask = torch.zeros((seq_length, seq_length), device=device)
        global_mask[:self.global_tokens, :] = 1
        global_mask[:, :self.global_tokens] = 1
        
        # Combine masks with learnable weights per head
        head_weights = torch.softmax(
            self.head_weights, dim=-1
        ).view(1, num_heads, 1, 1)
        
        combined_mask = (
            window_mask.unsqueeze(0) * head_weights[:, :, 0:1, :] +
            lsh_mask.unsqueeze(1) * head_weights[:, :, 1:2, :] +
            global_mask.unsqueeze(0).unsqueeze(1) * head_weights[:, :, 2:3, :]
        )
        
        # Apply sparsity threshold
        sparse_mask = (
            combined_mask > 
            torch.quantile(combined_mask, self.sparsity_factor, dim=-1, keepdim=True)
        ).float()
        
        return sparse_mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Use C++ kernel for attention computation
        output = attention_kernels.compute_sparse_attention(
            query=self.query(hidden_states),
            key=self.key(hidden_states),
            value=self.value(hidden_states),
            mask=attention_mask,
            dropout_prob=self.dropout.p
        )
        return output
