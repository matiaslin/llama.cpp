#include "common.cuh"

__global__ void paged_attention_write_kernel(
    const float * k_new,        // [head_dim, n_heads_kv, batch_size]
    const float * v_new,        // [head_dim, n_heads_kv, batch_size]
    half *        kv_cache,     // The paged cache
    const int *   write_slots,  // Global slot index for each token
    const int *   batch_offsets,
    const int *   batch_lens,
    const size_t  cache_stride_token,    // KV cache: elements between tokens in a block
    const size_t  cache_stride_head,     // KV cache: elements between heads
    const size_t  cache_stride_block,    // KV cache: elements between physical blocks
    const size_t  k_input_stride_token,  // K input: elements between tokens
    const size_t  k_input_stride_head,   // K input: elements between heads
    const size_t  v_input_stride_token,  // V input: elements between tokens
    const size_t  v_input_stride_head,   // V input: elements between heads
    const int     n_heads_kv,
    const int     block_size);

__global__ void paged_attention_decode_kernel(const float * __restrict__ q,
                                              const half * __restrict__ kv_cache,
                                              const int * __restrict__ block_table,
                                              const int * __restrict__ context_lens,
                                              const int * __restrict__ batch_offsets,
                                              const int * __restrict__ batch_lens,
                                              const size_t cache_stride_token,
                                              const size_t cache_stride_head,
                                              const size_t cache_stride_block,
                                              const size_t q_input_stride_token,
                                              const size_t q_input_stride_head,
                                              const int    n_heads_kv,
                                              const int    block_size,
                                              const int    max_blocks,
                                              const float  scale,
                                              float * __restrict__ out);

void ggml_cuda_op_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
