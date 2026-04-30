#include "pagedattn.cuh"

__device__ __forceinline__ float block_reduce_sum_full(float val, float * __restrict__ smem, int tid, int head_dim) {
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = (head_dim + 31) >> 5;

    // warp-level reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    // Each warp's lane 0 depositits partial sum into shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces the per-warp partial
    float warp_val = (tid < n_warps) ? smem[tid] : 0.0f;
    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_val += __shfl_down_sync(0xffffffffu, warp_val, offset);
        }
        if (lane == 0) {
            smem[0] = warp_val;
        }
    }
    __syncthreads();
    return smem[0];  // this will be identical in every thread
}

__global__ void paged_attention_write_kernel(
    const float * __restrict__ k_new,      // [head_dim, n_heads_kv, batch_size]
    const float * __restrict__ v_new,      // [head_dim, n_heads_kv, batch_size]
    half * __restrict__ kv_cache,          // The paged cache
    const int * __restrict__ write_slots,  // Global slot index for each token
    const int * __restrict__ batch_offsets,
    const int * __restrict__ batch_lens,
    const size_t cache_stride_token,    // KV cache: elements between tokens in a block
    const size_t cache_stride_head,     // KV cache: elements between heads
    const size_t cache_stride_block,    // KV cache: elements between physical blocks
    const size_t k_input_stride_token,  // K input: elements between tokens (nb[2]/sizeof)
    const size_t k_input_stride_head,   // K input: elements between heads (nb[1]/sizeof)
    const size_t v_input_stride_token,  // V input: same concept as K
    const size_t v_input_stride_head,   // V input: same concept as K
    const int    n_heads_kv,
    const int    block_size) {
    const int head_idx = blockIdx.x;   // 0 to n_heads_kv - 1
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;  // 0 to head_dim - 1

    const int seq_start  = batch_offsets[seq_idx];
    const int num_tokens = batch_lens[seq_idx];

    for (int i = 0; i < num_tokens; ++i) {
        const int token_batch_idx = seq_start + i;
        const int target_slot     = write_slots[token_batch_idx];

        // Map slot to block and internal offset
        const int block_id       = target_slot / block_size;
        const int token_in_block = target_slot % block_size;

        // K is at head_idx, V is at n_heads_kv + head_idx
        const size_t k_cache_idx = (size_t) block_id * cache_stride_block + (size_t) head_idx * cache_stride_head +
                                   (size_t) token_in_block * cache_stride_token + tid;
        const size_t v_cache_idx = (size_t) block_id * cache_stride_block +
                                   (size_t) (n_heads_kv + head_idx) * cache_stride_head +
                                   (size_t) token_in_block * cache_stride_token + tid;

        // Input offsets honor strides so views into a fused QKV tensor work without ggml_cont.
        const size_t k_input_off =
            (size_t) token_batch_idx * k_input_stride_token + (size_t) head_idx * k_input_stride_head + tid;
        const size_t v_input_off =
            (size_t) token_batch_idx * v_input_stride_token + (size_t) head_idx * v_input_stride_head + tid;

        kv_cache[k_cache_idx] = __float2half(k_new[k_input_off]);
        kv_cache[v_cache_idx] = __float2half(v_new[v_input_off]);
    }
}

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
                                              float * __restrict__ out) {
    extern __shared__ float smem[];

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;

    const int n_heads  = gridDim.x;
    const int head_dim = blockDim.x;

    const int kv_head_idx = head_idx / (n_heads / n_heads_kv);

    const int seq_start      = batch_offsets[seq_idx];
    const int num_new_tokens = batch_lens[seq_idx];

    for (int i = 0; i < num_new_tokens; i++) {
        const int token_batch_idx = seq_start + i;

        const size_t q_off =
            (size_t) token_batch_idx * q_input_stride_token + (size_t) head_idx * q_input_stride_head + tid;
        float q_val = q[q_off] * scale;

        float qk_max  = -FLT_MAX;
        float exp_sum = 0.0f;
        float acc     = 0.0f;

        const int ctx_len    = context_lens[seq_idx];
        const int q_pos      = (ctx_len - num_new_tokens) + i;
        const int num_blocks = (q_pos / block_size) + 1;

        for (int bid = 0; bid < num_blocks; bid++) {
            const int physical_block = block_table[seq_idx * max_blocks + bid];
            const int start_token    = bid * block_size;
            const int end_token      = min(start_token + block_size, q_pos + 1);

            for (int token = start_token; token < end_token; ++token) {
                const int token_in_block = token % block_size;

                const size_t k_idx = tid + token_in_block * cache_stride_token + kv_head_idx * cache_stride_head +
                                     physical_block * cache_stride_block;

                const size_t v_idx = tid + token_in_block * cache_stride_token +
                                     (n_heads_kv + kv_head_idx) * cache_stride_head +
                                     physical_block * cache_stride_block;

                float k_val = __half2float(kv_cache[k_idx]);
                float v_val = __half2float(kv_cache[v_idx]);

                // Calculate full dot product and return the same scalar in every thread
                const float qk = block_reduce_sum_full(q_val * k_val, smem, tid, head_dim);

                // Online softmax update
                const float qk_max_new = fmaxf(qk_max, qk);
                const float exp_old    = __expf(qk_max - qk_max_new);
                const float exp_new    = __expf(qk - qk_max_new);

                exp_sum = exp_sum * exp_old + exp_new;
                acc     = acc * exp_old + exp_new * v_val;
                qk_max  = qk_max_new;
            }
        }

        const int out_idx = (size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim + tid;

        out[out_idx] = acc / (exp_sum + 1e-6f);
    }
}

void ggml_cuda_op_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_new         = dst->src[1];
    const ggml_tensor * v_new         = dst->src[2];
    const ggml_tensor * kv_cache      = dst->src[3];  // KV interleaved layout
    const ggml_tensor * block_table   = dst->src[5];
    const ggml_tensor * write_slots   = dst->src[6];
    const ggml_tensor * context_lens  = dst->src[7];
    const ggml_tensor * batch_offsets = dst->src[8];
    const ggml_tensor * batch_lens    = dst->src[9];

    const float * op_params_f = (const float *) (dst->op_params);
    const float   scale       = op_params_f[0];
    const int     block_size  = ((const int32_t *) (op_params_f + 1))[0];
    const int     max_blocks  = ((const int32_t *) (op_params_f + 2))[0];

    const int head_dim   = q->ne[0];
    const int n_heads    = q->ne[1];
    const int n_seq      = batch_lens->ne[0];
    const int n_heads_kv = k_new->ne[1];

    GGML_ASSERT(n_heads != 0 && "n_head cannot be 0.");
    GGML_ASSERT(n_heads_kv != 0 && "n_heads_kv cannot be 0.");
    GGML_ASSERT(head_dim <= 1024 && "head_dim exceeds maximum supported (1024)");
    GGML_ASSERT(n_heads % n_heads_kv == 0 && "n_heads must be divisible by n_heads_kv");

    // Extracting strides
    const size_t cache_stride_token = kv_cache->nb[1] / sizeof(half);
    const size_t cache_stride_head  = kv_cache->nb[2] / sizeof(half);
    const size_t cache_stride_block = kv_cache->nb[3] / sizeof(half);

    // Input strides honor source layout
    const size_t q_input_stride_token = q->nb[2] / sizeof(float);
    const size_t q_input_stride_head  = q->nb[1] / sizeof(float);
    const size_t k_input_stride_token = k_new->nb[2] / sizeof(float);
    const size_t k_input_stride_head  = k_new->nb[1] / sizeof(float);
    const size_t v_input_stride_token = v_new->nb[2] / sizeof(float);
    const size_t v_input_stride_head  = v_new->nb[1] / sizeof(float);

    dim3 block_dims(head_dim);       // one thread per dimension of head
    dim3 grid_dims(n_heads, n_seq);  // one block per head per sequence

    // Write kernel - Grid (n_heads_kv, n_seq), Block (head_dim)
    paged_attention_write_kernel<<<dim3(n_heads_kv, n_seq), dim3(head_dim), 0, ctx.stream()>>>(
        (const float *) k_new->data, (const float *) v_new->data, (half *) kv_cache->data,
        (const int *) write_slots->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
        cache_stride_token, cache_stride_head, cache_stride_block, k_input_stride_token, k_input_stride_head,
        v_input_stride_token, v_input_stride_head, n_heads_kv, block_size);

    // Shared memory
    const size_t n_warps    = ((size_t) head_dim + 31) / 32;
    const size_t smem_bytes = n_warps * sizeof(float);

    // Manually request extended shared memory if needed (>48 KB)
    // https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
    if (smem_bytes > 48 * 1024) {
        GGML_ASSERT(smem_bytes <= 96 * 1024 && "smem exceeds 96KB limit");
        CUDA_CHECK(cudaFuncSetAttribute(paged_attention_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int) smem_bytes));
    }

    // Read kernel - Grid (n_heads_kv, n_seq), Block (head_dim)
    paged_attention_decode_kernel<<<dim3(n_heads, n_seq), dim3(head_dim), smem_bytes, ctx.stream()>>>(
        (const float *) q->data, (const half *) kv_cache->data, (const int *) block_table->data,
        (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
        cache_stride_token, cache_stride_head, cache_stride_block, q_input_stride_token, q_input_stride_head,
        n_heads_kv, block_size, max_blocks, scale, (float *) dst->data);
}
