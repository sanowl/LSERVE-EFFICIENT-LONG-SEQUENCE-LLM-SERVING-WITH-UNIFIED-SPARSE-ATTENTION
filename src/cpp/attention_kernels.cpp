#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

// Constants for performance optimization
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_SEQ_LENGTH = 16384;
constexpr int SHARED_MEM_SIZE = 48 * 1024;  // 48KB shared memory

// Helper functions for attention computation
__device__ __forceinline__ float scaled_dot_product(
    const float* query,
    const float* key,
    int head_dim,
    float scaling_factor
) {
    float dot_product = 0.0f;
    #pragma unroll
    for (int i = 0; i < head_dim; ++i) {
        dot_product += query[i] * key[i];
    }
    return dot_product * scaling_factor;
}

// Optimized sparse mask generation kernel
__global__ void generate_sparse_attention_mask_kernel(
    const float* scores,
    const int* bucket_ids,
    float* mask,
    int batch_size,
    int num_heads,
    int seq_length,
    int window_size,
    float sparsity_factor
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    
    // Shared memory for local computations
    __shared__ float local_scores[BLOCK_SIZE];
    __shared__ int local_buckets[BLOCK_SIZE];
    
    const int global_idx = batch_idx * (num_heads * seq_length * seq_length) +
                          head_idx * (seq_length * seq_length) +
                          bid * BLOCK_SIZE + tid;
                          
    if (tid < seq_length) {
        local_scores[tid] = scores[global_idx];
        local_buckets[tid] = bucket_ids[batch_idx * seq_length + tid];
    }
    __syncthreads();
    
    // Compute sliding window mask
    const int query_idx = bid * BLOCK_SIZE + tid;
    if (query_idx < seq_length) {
        for (int key_idx = 0; key_idx < seq_length; ++key_idx) {
            const int mask_idx = global_idx * seq_length + key_idx;
            
            // Sliding window condition
            bool in_window = abs(query_idx - key_idx) <= window_size / 2;
            
            // LSH bucket condition
            bool same_bucket = local_buckets[tid] == local_buckets[key_idx];
            
            // Combine conditions
            mask[mask_idx] = (in_window || same_bucket) ? 1.0f : 0.0f;
            
            // Apply sparsity threshold
            if (local_scores[tid] < sparsity_factor) {
                mask[mask_idx] = 0.0f;
            }
        }
    }
}

// Main sparse attention kernel with optimizations
__global__ void sparse_attention_forward_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ mask,
    float* __restrict__ output,
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
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    
    // Shared memory declarations
    extern __shared__ float shared_mem[];
    float* shared_q = shared_mem;
    float* shared_k = shared_mem + head_dim;
    float* shared_v = shared_k + head_dim;
    float* shared_scores = shared_v + head_dim;
    
    // Local accumulators
    float local_sum = 0.0f;
    float max_score = -INFINITY;
    
    // Get global position
    const int query_idx = bid * BLOCK_SIZE + tid;
    const int batch_offset = batch_idx * num_heads * seq_length * head_dim;
    const int head_offset = head_idx * seq_length * head_dim;
    
    // Load query into shared memory
    if (tid < head_dim && query_idx < seq_length) {
        const int q_idx = batch_offset + head_offset + query_idx * head_dim + tid;
        shared_q[tid] = query[q_idx];
    }
    __syncthreads();
    
    // Compute attention scores and apply mask
    float scores[BLOCK_SIZE];
    int valid_scores = 0;
    
    for (int key_block = 0; key_block < seq_length; key_block += BLOCK_SIZE) {
        // Load key and value block into shared memory
        if (tid < head_dim && (key_block + tid) < seq_length) {
            const int k_idx = batch_offset + head_offset + (key_block + tid) * head_dim;
            const int v_idx = k_idx;
            shared_k[tid] = key[k_idx];
            shared_v[tid] = value[v_idx];
        }
        __syncthreads();
        
        // Compute scores for this block
        if (query_idx < seq_length) {
            const float score = scaled_dot_product(
                shared_q,
                shared_k,
                head_dim,
                scaling_factor
            );
            
            // Apply mask if available
            if (mask != nullptr) {
                const int mask_idx = batch_idx * seq_length * seq_length +
                                   query_idx * seq_length + key_block + tid;
                scores[valid_scores] = score * mask[mask_idx];
            } else {
                scores[valid_scores] = score;
            }
            
            max_score = max(max_score, scores[valid_scores]);
            valid_scores++;
        }
        __syncthreads();
    }
    
    // Compute softmax and apply dropout
    float softmax_denom = 0.0f;
    for (int i = 0; i < valid_scores; ++i) {
        scores[i] = exp(scores[i] - max_score);
        softmax_denom += scores[i];
    }
    
    // Apply dropout using curand
    curandState local_state;
    curand_init(
        clock64(),
        query_idx,
        0,
        &local_state
    );
    
    // Compute weighted sum
    for (int i = 0; i < valid_scores; ++i) {
        const float dropout_mask = (curand_uniform(&local_state) > dropout_prob) ? 1.0f : 0.0f;
        scores[i] = (scores[i] * dropout_mask) / softmax_denom;
        
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            local_sum += scores[i] * shared_v[d];
        }
    }
    
    // Write output
    if (query_idx < seq_length) {
        const int out_idx = batch_offset + head_offset + query_idx * head_dim;
        output[out_idx] = local_sum;
    }
}

// LSH bucketing implementation
__global__ void compute_lsh_buckets_kernel(
    const float* __restrict__ input,
    int* __restrict__ buckets,
    const float* __restrict__ random_rotations,
    const int batch_size,
    const int seq_length,
    const int hidden_size,
    const int num_hashes,
    const int num_buckets
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_rotations = shared_mem + hidden_size;
    
    // Load input and random rotations into shared memory
    if (tid < hidden_size) {
        shared_input[tid] = input[batch_idx * seq_length * hidden_size + bid * hidden_size + tid];
        shared_rotations[tid] = random_rotations[tid];
    }
    __syncthreads();
    
    // Compute hash values
    float hash_value = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        hash_value += shared_input[i] * shared_rotations[i];
    }
    
    // Convert to bucket index
    if (bid < seq_length) {
        const int bucket_idx = (static_cast<int>(hash_value * num_buckets) + num_buckets) % num_buckets;
        buckets[batch_idx * seq_length + bid] = bucket_idx;
    }
}

// Python interface implementations
torch::Tensor compute_sparse_attention(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask,
    float dropout_prob
) {
    // Get tensor dimensions
    const int batch_size = query.size(0);
    const int num_heads = query.size(1);
    const int seq_length = query.size(2);
    const int head_dim = query.size(3);
    
    // Allocate output tensor
    auto output = torch::empty_like(value);
    
    // Calculate launch parameters
    const dim3 blocks(
        (seq_length + BLOCK_SIZE - 1) / BLOCK_SIZE,
        num_heads,
        batch_size
    );
    const dim3 threads(BLOCK_SIZE);
    const int shared_mem_size = 3 * head_dim * sizeof(float);
    
    // Launch kernel
    sparse_attention_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        mask.defined() ? mask.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_length,
        head_dim,
        1.0f / sqrt(head_dim),
        dropout_prob
    );
    
    return output;
}

torch::Tensor compute_lsh_buckets(
    torch::Tensor input,
    torch::Tensor random_rotations,
    int num_buckets
) {
    const int batch_size = input.size(0);
    const int seq_length = input.size(1);
    const int hidden_size = input.size(2);
    const int num_hashes = random_rotations.size(0);
    
    auto buckets = torch::empty(
        {batch_size, seq_length},
        torch::dtype(torch::kInt32).device(input.device())
    );
    
    const dim3 blocks(seq_length, batch_size);
    const dim3 threads(BLOCK_SIZE);
    const int shared_mem_size = (hidden_size + hidden_size) * sizeof(float);
    
    compute_lsh_buckets_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        buckets.data_ptr<int>(),
        random_rotations.data_ptr<float>(),
        batch_size,
        seq_length,
        hidden_size,
        num_hashes,
        num_buckets
    );
    
    return buckets;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_sparse_attention", &compute_sparse_attention);
    m.def("compute_lsh_buckets", &compute_lsh_buckets);
}
