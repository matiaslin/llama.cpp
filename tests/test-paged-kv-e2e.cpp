// tests/test-paged-kv-e2e.cpp
//
// End-to-end equivalence test for paged KV cache.
// We compare top-K agreement rather than raw logit values because the paged
// attention path uses a custom CUDA kernel with online softmax, while the
// unified path uses standard ggml attention with two-pass softmax. The two
// produce mathematically equivalent results but with different F16
// accumulation order, drifts on the order of 0.05-0.5 in raw
// logit values is expected and not a correctness issue. Top-K set agreement
// is robust to this drift while still catching the real issues (e.g. cross-device
// reads, layout corruption, MQA broadcast bugs) which produce wildly
// different distributions.
//
// Also samples N_COMPARE tokens greedy as a secondary 'cheap' check.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#define EXPECT_TRUE(x)                                                   \
    do {                                                                 \
        if (!(x)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            throw std::runtime_error("FAILED assertion.");               \
        }                                                                \
    } while (0)

static constexpr const char * TEST_PROMPT       = "Once upon a time there was a lovely";
static constexpr int          N_PREDICT         = 16;
static constexpr int          N_COMPARE         = 4;  // token-equivalence window
static constexpr int          TOP_K             = 5;
static constexpr int          MIN_TOP_K_OVERLAP = 4;  // at least 4 of top-5 must match

// Result of running one path: prefill-final logits + sampled token sequence.
struct path_result {
    std::vector<float>       prefill_logits;  // [n_vocab]
    std::vector<llama_token> tokens;          // [N_PREDICT]
    int                      n_vocab = 0;
};

static path_result run_non_paged(const std::string & model_path) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 256;
    params.n_batch       = 64;
    params.n_ubatch      = 64;
    params.n_predict     = N_PREDICT;
    params.sampling.temp = 0.0f;  // greedy
    params.warmup        = false;
    params.kv_paged      = false;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, TEST_PROMPT, true);
    EXPECT_TRUE(!prompt_tokens.empty());

    // Prefill
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    EXPECT_TRUE(llama_decode(ctx, batch) == 0);

    // Capture prefill-final logits BEFORE any further decode steps overwrite them.
    path_result result;
    result.n_vocab = n_vocab;
    {
        const float * raw = llama_get_logits_ith(ctx, -1);  // last logit
        EXPECT_TRUE(raw != nullptr);
        result.prefill_logits.assign(raw, raw + n_vocab);
    }

    // Sample N_PREDICT tokens for the secondary token-equivalence check.
    common_sampler * smpl = common_sampler_init(model, params.sampling);
    EXPECT_TRUE(smpl != nullptr);

    llama_token cur = -1;
    for (int i = 0; i < N_PREDICT; ++i) {
        cur = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, cur, true);
        result.tokens.push_back(cur);
        if (llama_vocab_is_eog(vocab, cur)) {
            break;
        }

        llama_batch step = llama_batch_get_one(&cur, 1);
        EXPECT_TRUE(llama_decode(ctx, step) == 0);
    }

    common_sampler_free(smpl);
    return result;
}

static path_result run_paged(const std::string & model_path) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 256;
    params.n_batch       = 64;
    params.n_ubatch      = 64;
    params.n_predict     = N_PREDICT;
    params.sampling.temp = 0.0f;  // greedy
    params.warmup        = false;
    params.kv_paged      = true;
    params.n_gpu_blocks  = 64;
    params.n_cpu_blocks  = 16;
    params.n_sequences   = 1;
    params.n_parallel    = 1;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_paged_scheduler * sched = llama_paged_scheduler_init(ctx);
    EXPECT_TRUE(sched != nullptr);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, TEST_PROMPT, true);
    EXPECT_TRUE(!prompt_tokens.empty());

    bool ok = llama_paged_scheduler_add_request(sched, prompt_tokens.data(), prompt_tokens.size(), 0);
    EXPECT_TRUE(ok);

    common_sampler * smpl = common_sampler_init(model, params.sampling);
    EXPECT_TRUE(smpl != nullptr);

    path_result result;
    result.n_vocab                      = n_vocab;
    bool        captured_prefill_logits = false;
    llama_batch batch                   = {};

    while ((int) result.tokens.size() < N_PREDICT) {
        bool prepared = llama_paged_scheduler_prepare_batch(sched, &batch);
        EXPECT_TRUE(prepared);
        if (batch.n_tokens == 0) {
            break;
        }

        EXPECT_TRUE(llama_decode(ctx, batch) == 0);
        llama_synchronize(ctx);

        const llama_paged_batch_info * info = llama_paged_scheduler_get_batch_info(sched);
        EXPECT_TRUE(info != nullptr && info->n_seq == 1);

        const int32_t last_idx = info->batch_offsets[0] + info->batch_lens[0] - 1;

        // First decode is the prefill — capture its final logits before
        // sampling anything else.
        if (!captured_prefill_logits) {
            const float * raw = llama_get_logits_ith(ctx, last_idx);
            EXPECT_TRUE(raw != nullptr);
            result.prefill_logits.assign(raw, raw + n_vocab);
            captured_prefill_logits = true;
        }

        llama_token next = common_sampler_sample(smpl, ctx, last_idx);
        common_sampler_accept(smpl, next, true);
        result.tokens.push_back(next);

        bool   stop      = llama_vocab_is_eog(vocab, next) || (int) result.tokens.size() >= N_PREDICT;
        int8_t stop_flag = stop ? 1 : 0;
        llama_paged_scheduler_update(sched, &batch, &next, &stop_flag);
        if (stop) {
            break;
        }
    }

    common_sampler_free(smpl);
    llama_paged_scheduler_free(sched);
    return result;
}

// Two-request paged run where B is queued only after A has prefilled and
// registered its prompt blocks. B then claims A's prompt blocks via the
// prefix-sharing path and serves only the unshared tail through the model.
struct paired_paged_result {
    path_result a_res;
    path_result b_res;
    bool        prefix_shared      = false;  // sanity: B was admitted with a non-empty match
    int32_t     b_first_batch_lens = -1;     // B's batch_lens on its prefill step
    int32_t     b_first_n_past     = -1;     // B's n_past at first appearance
    size_t      n_prompt_tokens    = 0;
};

static paired_paged_result run_paged_with_shared_prefix(const std::string & model_path) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 256;
    params.n_batch       = 64;
    params.n_ubatch      = 64;
    params.n_predict     = N_PREDICT;
    params.sampling.temp = 0.0f;
    params.warmup        = false;
    params.kv_paged      = true;
    params.block_size    = 4;  // small block to ensure the short test prompt covers > 1 full block
    params.n_gpu_blocks  = 64;
    params.n_cpu_blocks  = 16;
    params.n_sequences   = 2;
    params.n_parallel    = 2;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_paged_scheduler * sched = llama_paged_scheduler_init(ctx);
    EXPECT_TRUE(sched != nullptr);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, TEST_PROMPT, true);
    EXPECT_TRUE(!prompt_tokens.empty());
    EXPECT_TRUE(prompt_tokens.size() >= (size_t) (params.block_size * 2));

    EXPECT_TRUE(llama_paged_scheduler_add_request(sched, prompt_tokens.data(), prompt_tokens.size(), 0));

    common_sampler * smpl_a = common_sampler_init(model, params.sampling);
    common_sampler * smpl_b = common_sampler_init(model, params.sampling);
    EXPECT_TRUE(smpl_a != nullptr && smpl_b != nullptr);

    paired_paged_result out;
    out.a_res.n_vocab   = n_vocab;
    out.b_res.n_vocab   = n_vocab;
    out.n_prompt_tokens = prompt_tokens.size();

    bool        b_queued           = false;
    bool        captured_a_prefill = false;
    bool        captured_b_prefill = false;
    llama_batch batch              = {};

    while ((int) out.a_res.tokens.size() < N_PREDICT || (int) out.b_res.tokens.size() < N_PREDICT) {
        // Queue B only after A's first prefill has run, so A's prompt blocks
        // are registered and findable via the prefix-match path.
        if (!b_queued && captured_a_prefill) {
            EXPECT_TRUE(llama_paged_scheduler_add_request(sched, prompt_tokens.data(), prompt_tokens.size(), 1));
            b_queued = true;
        }

        bool prepared = llama_paged_scheduler_prepare_batch(sched, &batch);
        EXPECT_TRUE(prepared);
        if (batch.n_tokens == 0) {
            break;
        }

        EXPECT_TRUE(llama_decode(ctx, batch) == 0);
        llama_synchronize(ctx);

        const llama_paged_batch_info * info = llama_paged_scheduler_get_batch_info(sched);
        EXPECT_TRUE(info != nullptr);

        std::vector<llama_token> sampled(info->n_seq, 0);
        std::vector<int8_t>      stop_flags(info->n_seq, 0);

        for (int sid = 0; sid < info->n_seq; ++sid) {
            const int32_t off      = info->batch_offsets[sid];
            const int32_t len      = info->batch_lens[sid];
            const int32_t last_idx = off + len - 1;
            const int32_t req_id   = batch.seq_id[off][0];

            const float * raw = llama_get_logits_ith(ctx, last_idx);
            EXPECT_TRUE(raw != nullptr);

            path_result &    dst      = (req_id == 0) ? out.a_res : out.b_res;
            common_sampler * smp      = (req_id == 0) ? smpl_a : smpl_b;
            bool &           captured = (req_id == 0) ? captured_a_prefill : captured_b_prefill;

            if (!captured) {
                dst.prefill_logits.assign(raw, raw + n_vocab);
                captured = true;
                if (req_id == 1) {
                    out.b_first_batch_lens = len;
                    // n_past at the start of this step (populate_batch_from set pos values)
                    out.b_first_n_past     = batch.pos[off];
                    out.prefix_shared      = (len < (int32_t) prompt_tokens.size());
                }
            }

            if ((int) dst.tokens.size() < N_PREDICT) {
                llama_token next = common_sampler_sample(smp, ctx, last_idx);
                common_sampler_accept(smp, next, true);
                dst.tokens.push_back(next);
                sampled[sid]    = next;
                bool stop       = llama_vocab_is_eog(vocab, next) || (int) dst.tokens.size() >= N_PREDICT;
                stop_flags[sid] = stop ? 1 : 0;
            } else {
                stop_flags[sid] = 1;
            }
        }

        llama_paged_scheduler_update(sched, &batch, sampled.data(), stop_flags.data());
    }

    common_sampler_free(smpl_a);
    common_sampler_free(smpl_b);
    llama_paged_scheduler_free(sched);
    return out;
}

static void compare_results(const path_result & ref, const path_result & paged) {
    auto top_k = [](const std::vector<float> & l, int k) {
        std::vector<int> idx(l.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&l](int a, int b) { return l[a] > l[b]; });
        idx.resize(k);
        return idx;
    };

    const auto top_ref   = top_k(ref.prefill_logits, TOP_K);
    const auto top_paged = top_k(paged.prefill_logits, TOP_K);

    // Argmax must match: the most-confident next token should be identical.
    EXPECT_TRUE(top_ref[0] == top_paged[0]);

    // Top-K set overlap: at least MIN_TOP_K_OVERLAP of the K most likely
    // tokens must appear in both distributions.
    std::set<int> ref_set(top_ref.begin(), top_ref.end());
    int           overlap = 0;
    for (int t : top_paged) {
        if (ref_set.count(t)) {
            overlap++;
        }
    }

    fprintf(stderr, "test-paged-kv-e2e: top-%d argmax match: ref=%d paged=%d\n", TOP_K, top_ref[0], top_paged[0]);
    fprintf(stderr, "test-paged-kv-e2e: top-%d set overlap: %d/%d (require >= %d)\n", TOP_K, overlap, TOP_K,
            MIN_TOP_K_OVERLAP);

    if (overlap < MIN_TOP_K_OVERLAP) {
        fprintf(stderr,
                "FAIL: top-%d distributions diverge too much. Only %d of %d most-likely "
                "tokens match between ref and paged. Real correctness issue likely.\n",
                TOP_K, overlap, TOP_K);
        fprintf(stderr, "  ref:   ");
        for (int t : top_ref) {
            fprintf(stderr, "%d(%.3f) ", t, ref.prefill_logits[t]);
        }
        fprintf(stderr, "\n  paged: ");
        for (int t : top_paged) {
            fprintf(stderr, "%d(%.3f) ", t, paged.prefill_logits[t]);
        }
        fprintf(stderr, "\n");
        throw std::runtime_error("FAILED test.");
    }

    // Token-level secondary check: first N_COMPARE tokens must match.
    // We don't compare beyond N_COMPARE because greedy sampling on small
    // models is sensitive to argmax tiebreakers, and minor floating-point
    // accumulation differences between the paged and non-paged paths can
    // flip individual tokens after a handful of decode steps.
    EXPECT_TRUE((int) ref.tokens.size() >= N_COMPARE);
    EXPECT_TRUE((int) paged.tokens.size() >= N_COMPARE);
    for (int i = 0; i < N_COMPARE; ++i) {
        if (ref.tokens[i] != paged.tokens[i]) {
            fprintf(stderr, "FAIL: token %d differs in the equivalence window: ref=%d paged=%d\n", i, ref.tokens[i],
                    paged.tokens[i]);
            throw std::runtime_error("FAILED test.");
        }
    }
    fprintf(stderr, "test-paged-kv-e2e: PASSED\n");
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PAGED)) {
        fprintf(stderr, "usage: %s -m <model>\n", argv[0]);
        return 1;
    }
    if (params.model.path.empty()) {
        fprintf(stderr, "skip: no --model provided\n");
        return 0;
    }

    common_init();
    llama_backend_init();

    fprintf(stderr, "test-paged-kv-e2e: running non-paged reference\n");
    path_result ref = run_non_paged(params.model.path);
    fprintf(stderr, "  got %zu tokens, %d-vocab logits\n", ref.tokens.size(), ref.n_vocab);

    fprintf(stderr, "test-paged-kv-e2e: running paged path\n");
    path_result paged = run_paged(params.model.path);
    fprintf(stderr, "  got %zu tokens, %d-vocab logits\n", paged.tokens.size(), paged.n_vocab);

    compare_results(ref, paged);

    fprintf(stderr, "test-paged-kv-e2e: running paged path with shared prefix (B claims A's blocks)\n");
    paired_paged_result shared = run_paged_with_shared_prefix(params.model.path);
    fprintf(stderr, "  prompt tokens: %zu, B first batch_lens: %d, B first n_past: %d, prefix_shared=%d\n",
            shared.n_prompt_tokens, shared.b_first_batch_lens, shared.b_first_n_past, shared.prefix_shared ? 1 : 0);

    EXPECT_TRUE(shared.prefix_shared);
    EXPECT_TRUE(shared.b_first_n_past > 0);
    EXPECT_TRUE(shared.b_first_batch_lens >= 1);
    EXPECT_TRUE((size_t) (shared.b_first_n_past + shared.b_first_batch_lens) == shared.n_prompt_tokens);

    fprintf(stderr, "test-paged-kv-e2e: comparing A (paged, no prefix share) vs reference\n");
    compare_results(ref, shared.a_res);
    fprintf(stderr, "test-paged-kv-e2e: comparing B (paged, served via shared prefix) vs reference\n");
    compare_results(ref, shared.b_res);

    llama_backend_free();
    return 0;
}
