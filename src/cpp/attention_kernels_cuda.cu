#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Constants for optimization
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

__device__ float atomicMax(float* address, float val) {
    int* address_as_int = (int*) address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(max(__int_as_float(assumed), val)));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void sparse_attention_forward_kernel(
    const float* query,
    const float* key,
    const float* value,
    const float* mask,
    float* output,
    float* attention_probs,
    const int batch_size,
    const int num_heads,
    const int seq_length,
    const int head_dim,
    const float scaling_factor,
    const float dropout_prob
) {
    // Get block and thread indices
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    
    // Shared memory for query and key
    extern __shared__ float shared_memory[];
    float* shared_query = shared_memory;
    float* shared_key = shared_memory + BLOCK_SIZE;
    
    // Compute attention scores for this block
    for (int i = 0; i < seq_length; i += BLOCK_SIZE) {
        // Load query and key into shared memory
        if (i + tid < seq_length) {
            shared_query[tid] = query[
                ((batch_idx * num_heads + head_idx) * seq_length + i + tid) * head_dim
            ];
            shared_key[tid] = key[
                ((batch_idx * num_heads + head_idx) * seq_length + i + tid) * head_dim
            ];
        }
        __syncthreads();
        
        // Compute attention scores
        float score = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            score += shared_query[tid] * shared_key[j];
        }
        score *= scaling_factor;
        
        // Apply mask if provided
        if (mask != nullptr) {
            score += mask[batch_idx * seq_length + i + tid];
        }
        
        // Store attention score
        attention_probs[
            ((batch_idx * num_heads + head_idx) * seq_length + i + tid)
        ] = score;
        
        __syncthreads();
    }
}

__global__ void sparse_attention_backward_kernel(
    const float* grad_output,
    const float* attention_probs,
    const float* value,
    float* grad_query,
    float* grad_key,
    float* grad_value,
    const int batch_size,
    const int num_heads,
    const int seq_length,
    const int head_dim
) {
    // Get thread indices
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    
    // Shared memory declarations
    extern __shared__ float shared_memory[];
    float* shared_grad = shared_memory;
    float* shared_probs = shared_memory + BLOCK_SIZE;
    
    // Compute gradients
    for (int i = 0; i < seq_length; i += BLOCK_SIZE) {
        if (i + tid < seq_length) {
            shared_grad[tid] = grad_output[
                ((batch_idx * num_heads + head_idx) * seq_length + i + tid) * head_dim
            ];
            shared_probs[tid] = attention_probs[
                ((batch_idx * num_heads + head_idx) * seq_length + i + tid)
            ];
        }
        __syncthreads();
        
        // Compute gradients for query, key, and value
        float grad_q = 0.0f, grad_k = 0.0f, grad_v = 0.0f;
        
        for (int j = 0; j < head_dim; j++) {
            grad_q += shared_grad[tid] * value[
                ((batch_idx * num_heads + head_idx) * seq_length + j) * head_dim
            ] * shared_probs[j];
            
            grad_k += shared_grad[j] * value[
                ((batch_idx * num_heads + head_idx) * seq_length + tid) * head_dim
            ] * shared_probs[tid];
            
            grad_v += shared_grad[tid] * shared_probs[j];
        }
        
        // Store gradients
        grad_query[
            ((batch_idx * num_heads + head_idx) * seq_length + i + tid) * head_dim
        ] = grad_q;
        
        grad_key[
            ((batch_idx * num_heads + head_idx) * seq_length + i + tid) * head_dim
        ] = grad_k;
        
        grad_value[
            ((batch_idx * num_heads + head_idx) * seq_length + i + tid) * head_dim
        ] = grad_v;
        
        __syncthreads();
    }
}

// Launch parameters calculation
extern "C" void launch_sparse_attention_forward(
    const float* query,
    const float* key,
    const float* value,
    const float* mask,
    float* output,
    float* attention_probs,
    const int batch_size,
    const int num_heads,
    const int seq_length,
    const int head_dim,
    cudaStream_t stream
) {
    const dim3 blocks(
        (seq_length + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size,
        num_heads
    );
    const dim3 threads(BLOCK_SIZE);
    
    const int shared_memory_size = 2 * BLOCK_SIZE * sizeof(float);
    const float scaling_factor = 1.0f / sqrt(head_dim);
    
    sparse_attention_forward_kernel<<<blocks, threads, shared_memory_size, stream>>>(
        query, key, value, mask, output, attention_probs,
        batch_size, num_heads, seq_length, head_dim,
        scaling_factor, 0.1f  // dropout_prob
    );
}
