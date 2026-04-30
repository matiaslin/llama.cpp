#include "ggml-backend.h"
#include "llama-block-manager.h"
#include "llama-kv-cache-paged.h"
#include "llama-paged-scheduler-impl.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#define TEST(name) static void name()
#define RUN(name)                                   \
    do {                                            \
        fprintf(stderr, "  running %-40s ", #name); \
        fflush(stderr);                             \
        name();                                     \
        fprintf(stderr, "OK\n");                    \
    } while (0)

#define EXPECT_EQ(a, b)                                                                                          \
    do {                                                                                                         \
        auto _a = (a);                                                                                           \
        auto _b = (b);                                                                                           \
        if (_a != _b) {                                                                                          \
            fprintf(stderr, "\n  FAIL %s:%d: expected %s == %s, got %lld vs %lld\n", __FILE__, __LINE__, #a, #b, \
                    (long long) _a, (long long) _b);                                                             \
            std::abort();                                                                                        \
        }                                                                                                        \
    } while (0)

#define EXPECT_TRUE(x)                                                       \
    do {                                                                     \
        if (!(x)) {                                                          \
            fprintf(stderr, "\n  FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            std::abort();                                                    \
        }                                                                    \
    } while (0)
#define EXPECT_FALSE(x) EXPECT_TRUE(!(x))

// Testing block_manager main functionality

TEST(test_block_manager_leak_simple) {
    const uint32_t n_gpu_blocks = 16;
    const uint32_t n_cpu_blocks = 8;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);

    auto gpu_ids = block_manager.checkout_gpu_blocks(10);
    EXPECT_EQ(gpu_ids.size(), 10u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), (n_gpu_blocks - 10u));

    auto cpu_ids = block_manager.checkout_cpu_blocks(5);
    EXPECT_EQ(cpu_ids.size(), 5u);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), (n_cpu_blocks - 5u));

    block_manager.release_gpu_blocks(gpu_ids);
    block_manager.release_cpu_blocks(cpu_ids);

    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);
}

// Testing that we always return to full (stress test)
TEST(test_block_manager_leak_repeated) {
    const uint32_t block_size   = 16;
    const uint32_t n_gpu_blocks = 64;
    const uint32_t n_cpu_blocks = 32;
    const float    watermark    = 0.0f;
    const int      n_iter       = 1000;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    for (int iter = 0; iter < n_iter; ++iter) {
        const uint32_t n   = (iter % block_size) + 1;
        auto           ids = block_manager.checkout_gpu_blocks(n);
        EXPECT_EQ(ids.size(), (size_t) n);
        block_manager.release_gpu_blocks(ids);
        EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    }
}

// Attempting to check-out more blocks than available. It should return empty.
TEST(test_block_manager_checkout_too_many) {
    const uint32_t n_gpu_blocks = 8;
    const uint32_t n_cpu_blocks = 4;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto ids = block_manager.checkout_gpu_blocks(9);
    EXPECT_EQ(ids.size(), 0u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);

    auto ids2 = block_manager.checkout_cpu_blocks(5);
    EXPECT_EQ(ids2.size(), 0u);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);
}

TEST(test_block_manager_watermark) {
    // watermark=0.2 means 2 blocks reserved as safety (always consider max gpu blocks)
    const uint32_t n_gpu_blocks = 10;
    const uint32_t n_cpu_blocks = 10;
    const float    watermark    = 0.2f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    // 10 free, safety=2
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(8));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(9));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(10));

    auto ids = block_manager.checkout_gpu_blocks(5);
    EXPECT_EQ(ids.size(), 5u);
    // 5 free, safety=2
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(3));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(4));

    block_manager.release_gpu_blocks(ids);
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(8));
}

TEST(test_block_manager_watermark_zero) {
    // watermark=0 means the entire pool is requestable.
    const uint32_t n_gpu_blocks = 10;
    const uint32_t n_cpu_blocks = 10;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(10));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(11));
}

TEST(test_block_manager_gpu_cpu_disjoint) {
    // Just making sure CPU and GPU blocks are disjoint
    const uint32_t n_gpu_blocks = 8;
    const uint32_t n_cpu_blocks = 4;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto gpu_ids = block_manager.checkout_gpu_blocks(8);
    auto cpu_ids = block_manager.checkout_cpu_blocks(4);

    for (auto id : gpu_ids) {
        EXPECT_TRUE(block_manager.is_gpu(id));
    }
    for (auto id : cpu_ids) {
        EXPECT_FALSE(block_manager.is_gpu(id));
    }

    block_manager.release_gpu_blocks(gpu_ids);
    block_manager.release_cpu_blocks(cpu_ids);
}

TEST(test_block_manager_shared_gpu_block) {
    const uint32_t n_gpu_blocks = 1;
    const uint32_t n_cpu_blocks = 1;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    block_manager.checkout_gpu_blocks(1);
    // Mimick another sequence that matched in prefix
    block_manager.increment_ref(0);  // there's only one GPU block, so we know the idx
    EXPECT_EQ(block_manager.get_ref_count(0), (uint32_t) 2);

    // Calling 'release' on block with id:0 will not free it because
    // its ref_count is 2. We only free when their ref_count reaches 0
    block_manager.release_gpu_blocks({ 0 });
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(1));

    // Releasing it a second time to reduce its ref_count to 0.
    block_manager.release_gpu_blocks({ 0 });
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(1));
}

TEST(test_block_manager_register_lookup_roundtrip) {
    const uint32_t n_gpu_blocks = 4;
    const uint32_t n_cpu_blocks = 2;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto           ids = block_manager.checkout_gpu_blocks(1);
    const uint32_t bid = ids[0];

    const uint64_t                 hash   = 0xDEADBEEFCAFEBABEULL;  // arbitrary hex
    const std::vector<llama_token> tokens = { 1, 2, 3, 4, 5 };

    block_manager.register_block_hash(bid, hash, tokens);
    EXPECT_EQ(block_manager.lookup_block_by_hash(hash, tokens), bid);
}

TEST(test_block_manager_lookup_collision_mismatched_tokens) {
    const uint32_t n_gpu_blocks = 4;
    const uint32_t n_cpu_blocks = 2;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto           ids = block_manager.checkout_gpu_blocks(1);
    const uint32_t bid = ids[0];

    const uint64_t                 hash     = 0xDEADBEEFCAFEBABEULL;  // arbitrary hex
    const std::vector<llama_token> tokens_a = { 1, 2, 3 };
    const std::vector<llama_token> tokens_b = { 1, 2, 4 };            // different tokens

    block_manager.register_block_hash(bid, hash, tokens_a);
    EXPECT_EQ(block_manager.lookup_block_by_hash(hash, tokens_b), llama_block_manager::INVALID_BLOCK_ID);
    EXPECT_EQ(block_manager.lookup_block_by_hash(hash, tokens_a), bid);
}

TEST(test_block_manager_lookup_miss) {
    const uint32_t n_gpu_blocks = 4;
    const uint32_t n_cpu_blocks = 2;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    const uint64_t                 hash   = 0xDEADBEEFCAFEBABEULL;  // arbitrary hex
    const std::vector<llama_token> tokens = { 1, 2, 3, 4, 5 };

    EXPECT_EQ(block_manager.lookup_block_by_hash(hash, tokens), llama_block_manager::INVALID_BLOCK_ID);
}

TEST(test_block_manager_release_clears_hash) {
    const uint32_t n_gpu_blocks = 2;
    const uint32_t n_cpu_blocks = 1;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto           ids = block_manager.checkout_gpu_blocks(1);
    const uint32_t bid = ids[0];

    const uint64_t                 hash   = 0xDEADBEEFCAFEBABEULL;  // arbitrary hex
    const std::vector<llama_token> tokens = { 1, 2, 3, 4, 5 };

    block_manager.register_block_hash(bid, hash, tokens);
    EXPECT_EQ(block_manager.lookup_block_by_hash(hash, tokens), bid);

    // Releasing the block (ref_count == 0)
    block_manager.release_gpu_blocks(ids);
    EXPECT_EQ(block_manager.lookup_block_by_hash(hash, tokens), llama_block_manager::INVALID_BLOCK_ID);

    auto re_ids = block_manager.checkout_gpu_blocks(1);
    EXPECT_EQ(re_ids.size(), 1u);
    block_manager.register_block_hash(re_ids[0], hash, tokens);
    EXPECT_EQ(block_manager.lookup_block_by_hash(hash, tokens), re_ids[0]);
}

TEST(test_block_manager_shared_block_keeps_hash_until_zero) {
    const uint32_t n_gpu_blocks = 2;
    const uint32_t n_cpu_blocks = 1;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto           ids = block_manager.checkout_gpu_blocks(1);
    const uint32_t bid = ids[0];

    // Simulating prefix-sharing
    block_manager.increment_ref(bid);
    EXPECT_EQ(block_manager.get_ref_count(bid), (uint32_t) 2);

    const uint64_t                 hash   = 0xDEADBEEFCAFEBABEULL;  // arbitrary hex
    const std::vector<llama_token> tokens = { 1, 2, 3, 4, 5 };
    block_manager.register_block_hash(bid, hash, tokens);

    // ref_count 2 to 1.
    block_manager.release_gpu_blocks(ids);
    EXPECT_EQ(block_manager.get_ref_count(bid), (uint32_t) 1);
    EXPECT_EQ(block_manager.lookup_block_by_hash(hash, tokens), bid);

    // ref_count 1 to 0.
    block_manager.release_gpu_blocks(ids);
    EXPECT_EQ(block_manager.lookup_block_by_hash(hash, tokens), llama_block_manager::INVALID_BLOCK_ID);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), (uint32_t) 2);
}

// Testing llama_kv_cache_paged book-keeping

// kv_cache_paged with arbitrary shape (we only care about the bookkeeping)
static llama_kv_cache_paged make_kv() {
    return llama_kv_cache_paged(
        /*head_dim=*/64,
        /*n_heads_kv=*/4,
        /*block_size=*/16,
        /*n_layers=*/2,
        /*n_ubatch=*/32,
        /*n_seq_max=*/8);
}

TEST(test_seq_pos_default_unknown) {
    auto kv = make_kv();
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_max(0), -1);
    EXPECT_EQ(kv.seq_pos_min(42), -1);
}

TEST(test_seq_pos_set_and_get) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_max_pos(0, 17);
    EXPECT_EQ(kv.seq_pos_min(0), 5);
    EXPECT_EQ(kv.seq_pos_max(0), 17);

    // Independent per seq_id.
    kv.set_seq_min_pos(1, 100);
    EXPECT_EQ(kv.seq_pos_min(1), 100);
    EXPECT_EQ(kv.seq_pos_min(0), 5);  // unchanged
}

TEST(test_seq_pos_overwrite) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_min_pos(0, 9);
    EXPECT_EQ(kv.seq_pos_min(0), 9);
}

TEST(test_seq_pos_seq_rm_removes) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_max_pos(0, 17);
    EXPECT_EQ(kv.seq_pos_min(0), 5);

    kv.seq_rm(0, 0, 0);
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_max(0), -1);
}

TEST(test_seq_pos_clear_removes_all) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_min_pos(1, 7);
    kv.set_seq_min_pos(2, 9);

    kv.clear(/*data=*/false);
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_min(1), -1);
    EXPECT_EQ(kv.seq_pos_min(2), -1);
}

TEST(test_seq_cp_shares_blocks) {
    const uint32_t block_size        = 16;
    const uint32_t prefix_num_blocks = 3;
    const uint32_t n_gpu_blocks      = 4;
    const uint32_t n_cpu_blocks      = 4;
    const float    watermark         = 0.0f;

    llama_sequence_group group1                               = llama_sequence_group{};
    llama_sequence_group group2                               = llama_sequence_group{};
    group1.block_table                                        = {};
    group2.block_table                                        = {};
    std::unordered_map<uint32_t, llama_sequence_group *> regs = {
        {0,  &group1},
        { 1, &group2}
    };
    auto cb = [&regs](uint32_t seq_id) -> llama_sequence_group * {
        if (regs.find(seq_id) == regs.end()) {
            return nullptr;
        }
        return regs[seq_id];
    };

    auto                  kv            = make_kv();
    llama_block_manager & block_manager = kv.get_block_manager();
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    // dst starts with one block
    auto dst_og_id = block_manager.checkout_gpu_blocks(1);
    group2.block_table.push_back(dst_og_id[0]);

    // checkout 3 blocks for src
    auto new_ids = block_manager.checkout_gpu_blocks(3);
    EXPECT_EQ(new_ids.size(), prefix_num_blocks);
    for (uint32_t i = 0; i < prefix_num_blocks; ++i) {
        group1.block_table.push_back(new_ids[i]);
    }
    EXPECT_EQ(regs[0]->block_table.size(), prefix_num_blocks);
    EXPECT_EQ(regs[1]->block_table.size(), (uint32_t) 1);

    kv.set_sequence_group_lookup(cb);
    kv.seq_cp(0, 1, 0, prefix_num_blocks * block_size);

    EXPECT_EQ(regs[0]->block_table.size(), prefix_num_blocks);
    EXPECT_EQ(regs[1]->block_table.size(), prefix_num_blocks);
    for (uint32_t i = 0; i < prefix_num_blocks; ++i) {
        EXPECT_EQ(regs[0]->block_table[i], regs[1]->block_table[i]);
    }
    // The dst block might have its max/min seq pos updated
    EXPECT_EQ(kv.seq_pos_min(1), (llama_pos) 0);
    EXPECT_EQ(kv.seq_pos_max(1), (llama_pos) ((prefix_num_blocks * block_size) - 1));

    block_manager.release_gpu_blocks(new_ids);
    // We still have the dst who is using the blocks
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), (uint32_t) 1);
    block_manager.release_gpu_blocks(new_ids);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
}

TEST(test_free_blocks_releases_to_pool) {
    // Initialize a KV cache with a real CPU backend (used as both "GPU" and CPU).
    // This is fine for testing
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    const uint32_t n_gpu_blocks = 16;
    const uint32_t n_cpu_blocks = 8;
    const float    watermark    = 0.0f;

    auto kv = make_kv();
    kv.init(/*backend_gpu=*/backend,
            /*backend_cpu=*/backend, GGML_TYPE_F16, n_gpu_blocks, n_cpu_blocks, watermark);

    // allocate() pulls from the GPU block pool. After releasing, the count
    // must return to the initial value.
    const uint32_t n_gpu_initial = kv.get_num_gpu_blocks();
    EXPECT_EQ(n_gpu_initial, n_gpu_blocks);

    llama_sequence_group group;
    group.request_id = 0;
    group.n_prompt   = 32;  // 2 blocks
    group.n_decoded  = 0;

    bool success = kv.allocate(/*num_tokens=*/group.n_prompt + group.n_decoded, group);
    EXPECT_TRUE(success);
    EXPECT_EQ(group.block_table.size(), 2u);

    // Allocating a second sequence further reduces the pool.
    llama_sequence_group group2;
    group2.request_id = 1;
    group2.n_prompt   = 48;  // 3 blocks
    group2.n_decoded  = 0;

    success = kv.allocate(group2.n_prompt + group2.n_decoded, group2);
    EXPECT_TRUE(success);
    EXPECT_EQ(group2.block_table.size(), 3u);

    // Free the first sequence: we release 2 blocks
    kv.free_blocks(group);
    EXPECT_EQ(group.block_table.size(), 0u);

    llama_sequence_group group3;
    group3.request_id = 2;
    group3.n_prompt   = 32;  // 2 blocks
    group3.n_decoded  = 0;
    success           = kv.allocate(group3.n_prompt + group3.n_decoded, group3);
    EXPECT_TRUE(success);
    EXPECT_EQ(group3.block_table.size(), 2u);

    // All blocks released
    kv.free_blocks(group2);
    kv.free_blocks(group3);

    llama_sequence_group group_full;
    group_full.request_id = 99;
    group_full.n_prompt   = n_gpu_blocks * 16;  // exactly n_gpu_blocks worth
    group_full.n_decoded  = 0;
    success               = kv.allocate(group_full.n_prompt + group_full.n_decoded, group_full);
    EXPECT_TRUE(success);
    EXPECT_EQ(group_full.block_table.size(), 16u);
    kv.free_blocks(group_full);

    ggml_backend_free(backend);
}

// Testing scheduler

// For easy testing and clean-up.
// KV cache paged + scheduler (must free CPU backend)
struct paged_test_fixture {
    ggml_backend_t                              backend = nullptr;
    std::unique_ptr<llama_kv_cache_paged>       kv;
    std::unique_ptr<llama_paged_scheduler_impl> sched;

    ~paged_test_fixture() {
        if (backend) {
            ggml_backend_free(backend);
        }
    }

    // rule of 5
    paged_test_fixture()                                       = default;
    paged_test_fixture(paged_test_fixture &&)                  = default;
    paged_test_fixture & operator=(paged_test_fixture &&)      = default;
    paged_test_fixture(const paged_test_fixture &)             = delete;
    paged_test_fixture & operator=(const paged_test_fixture &) = delete;
};

static paged_test_fixture make_fixture(uint32_t n_ctx        = 128,
                                       uint32_t block_size   = 16,
                                       uint32_t n_batch      = 64,
                                       uint32_t n_gpu_blocks = 4,
                                       uint32_t n_cpu_blocks = 2) {
    paged_test_fixture fixture;
    fixture.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(fixture.backend != nullptr);

    fixture.kv = std::unique_ptr<llama_kv_cache_paged>(new llama_kv_cache_paged(
        /*head_dim=*/64u,
        /*n_heads_kv=*/4u,
        /*block_size=*/block_size,
        /*n_layers=*/2u,
        /*n_ubatch=*/n_batch,
        /*n_seq_max=*/8u));
    fixture.kv->init(fixture.backend, fixture.backend, GGML_TYPE_F16, n_gpu_blocks, n_cpu_blocks, /*watermark=*/0.0f);

    fixture.sched = std::unique_ptr<llama_paged_scheduler_impl>(
        new llama_paged_scheduler_impl(n_ctx, block_size, n_batch, fixture.kv.get()));
    return fixture;
}

static llama_sequence_group make_group(int32_t request_id, uint32_t n_prompt) {
    llama_sequence_group group;
    group.request_id     = request_id;
    group.n_prompt       = n_prompt;
    group.n_decoded      = 0;
    group.n_past         = 0;
    group.t_arrival_time = request_id;  // control ordering based on request_id
    group.logical_seq.assign(n_prompt, /*dummy token=*/1);
    return group;
}

TEST(test_scheduler_no_deadlock_on_empty) {
    // No requests queued means no deadlock
    auto                   fixture = make_fixture();
    llama_batch            batch   = {};
    llama_scheduler_status status  = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 0);
}

TEST(test_scheduler_deadlock_oversize_waiting_request) {
    // There are 2 blocks, the waiting request needs 3.
    // Attempt to process prefill for this requets will fail and the request will remain in waiting (deadlock).
    auto fixture = make_fixture(128, 16, 64, /*n_gpu_blocks=*/2, /*n_cpu_blocks=*/1);
    auto group   = make_group(0, /*n_prompts=*/32);

    bool queued = fixture.sched->queue_request(group);
    EXPECT_TRUE(queued);

    llama_batch            batch  = {};
    llama_scheduler_status status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::DEADLOCK);

    // Next steps will continue remained deadlocked
    status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::DEADLOCK);
}

TEST(test_scheduler_rejects_oversized_prompt) {
    auto fixture = make_fixture(/*n_ctx=*/64, /*block_size=*/16, /*n_batch=*/128,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    bool queued = fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/64));
    EXPECT_FALSE(queued);

    queued = fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/128));
    EXPECT_FALSE(queued);

    // A request just under the limit is accepted.
    queued = fixture.sched->queue_request(make_group(/*id=*/2, /*n_prompt=*/63));
    EXPECT_TRUE(queued);
}

TEST(test_cow_writes_to_shared_block) {
    auto           fixture       = make_fixture(/*n_ctx=*/128, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/4, /*n_cpu_blocks=*/2);
    auto &         block_manager = fixture.kv->get_block_manager();
    auto           ids           = block_manager.checkout_gpu_blocks(1);
    const uint32_t shared_bid    = ids[0];

    // Simulating prefix-sharing
    block_manager.increment_ref(shared_bid);
    EXPECT_EQ(block_manager.get_ref_count(shared_bid), 2u);

    llama_sequence_group writer{};
    writer.request_id  = 0;
    writer.n_past      = 0;
    writer.block_table = { shared_bid };

    const size_t old_free_gpu_blocks = block_manager.n_free_gpu_blocks();
    bool         success = fixture.kv->try_cow_shared_blocks(writer, /*writer_start_pos=*/0, /*n_writer_tokens=*/1);
    EXPECT_TRUE(success);

    EXPECT_TRUE(writer.block_table[0] != shared_bid);
    EXPECT_EQ(block_manager.get_ref_count(shared_bid), 1u);
    EXPECT_EQ(block_manager.get_ref_count(writer.block_table[0]), 1u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), old_free_gpu_blocks - 1u);
}

TEST(test_cow_no_op_for_unshared_block) {
    auto   fixture       = make_fixture(128, 16, 64, /*n_gpu_blocks=*/4, /*n_cpu_blocks=*/1);
    auto & block_manager = fixture.kv->get_block_manager();

    auto           ids = block_manager.checkout_gpu_blocks(1);
    const uint32_t bid = ids[0];
    EXPECT_EQ(block_manager.get_ref_count(bid), 1u);

    const size_t old_free_gpu_blocks = block_manager.n_free_gpu_blocks();

    llama_sequence_group writer{};
    writer.request_id  = 0;
    writer.n_past      = 0;
    writer.block_table = { bid };

    bool success = fixture.kv->try_cow_shared_blocks(writer, /*writer_start_pos=*/0, /*n_writer_tokens=*/1);
    EXPECT_TRUE(success);

    EXPECT_EQ(writer.block_table[0], bid);
    EXPECT_EQ(block_manager.get_ref_count(bid), 1u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), old_free_gpu_blocks);
}

TEST(test_cow_fails_under_gpu_pressure) {
    auto   fixture       = make_fixture(128, 16, 64, /*n_gpu_blocks=*/1, /*n_cpu_blocks=*/2);
    auto & block_manager = fixture.kv->get_block_manager();

    auto ids = block_manager.checkout_gpu_blocks(1);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), 0u);

    const uint32_t shared_bid = ids[0];
    block_manager.increment_ref(shared_bid);
    EXPECT_EQ(block_manager.get_ref_count(shared_bid), 2u);

    llama_sequence_group writer{};
    writer.request_id  = 0;
    writer.n_past      = 0;
    writer.block_table = { shared_bid };

    bool success = fixture.kv->try_cow_shared_blocks(writer, /*writer_start_pos=*/0, /*n_writer_tokens=*/1);
    EXPECT_FALSE(success);

    // Early exit without corrupting the writer's state
    EXPECT_EQ(writer.block_table[0], shared_bid);
    EXPECT_EQ(block_manager.get_ref_count(shared_bid), 2u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), 0u);
}

TEST(test_register_prompt_blocks_full_blocks) {
    const uint32_t block_size   = 4;
    const uint32_t n_gpu_blocks = 8;
    const uint32_t n_cpu_blocks = 2;
    const float    watermark    = 0.0f;

    auto kv = llama_kv_cache_paged(
        /*head_dim=*/64, /*n_heads_kv=*/4, block_size,
        /*n_layers=*/2, /*n_ubatch=*/32, /*n_seq_max=*/8);
    auto & block_manager = kv.get_block_manager();
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    llama_sequence_group group{};
    group.request_id  = 0;
    group.n_prompt    = 8;  // 2 full blocks
    group.n_decoded   = 0;
    group.logical_seq = { 1, 2, 3, 4, 5, 6, 7, 8 };
    group.block_table = block_manager.checkout_gpu_blocks(2);
    EXPECT_EQ(group.block_table.size(), 2u);

    // Both full prompt blocks will be findable via their cumulative hash
    kv.register_prompt_blocks(group);
    kv.register_prompt_blocks(group);  // second call must be a no-op (idempotent)

    auto hashes = llama_kv_cache_paged::compute_prompt_block_hashes(group.logical_seq, block_size);
    EXPECT_EQ(hashes.size(), 2u);

    const std::vector<llama_token> b0_tokens = { 1, 2, 3, 4 };  // block 0
    const std::vector<llama_token> b1_tokens = { 5, 6, 7, 8 };  // block 1

    EXPECT_EQ(block_manager.lookup_block_by_hash(hashes[0], b0_tokens), group.block_table[0]);
    EXPECT_EQ(block_manager.lookup_block_by_hash(hashes[1], b1_tokens), group.block_table[1]);
}

TEST(test_register_prompt_blocks_partial_block_ignored) {
    const uint32_t block_size   = 4;
    const uint32_t n_gpu_blocks = 4;
    const uint32_t n_cpu_blocks = 2;
    const float    watermark    = 0.0f;

    auto   kv            = llama_kv_cache_paged(64, 4, block_size, 2, 32, 8);
    auto & block_manager = kv.get_block_manager();
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    llama_sequence_group group{};
    group.request_id  = 0;
    group.n_prompt    = 6;                                     // 1 full block + 2 trailing tokens
    group.logical_seq = { 1, 2, 3, 4, 5, 6 };
    group.block_table = block_manager.checkout_gpu_blocks(2);  // 2 blocks held, but only 1 should register

    kv.register_prompt_blocks(group);

    auto hashes = llama_kv_cache_paged::compute_prompt_block_hashes(group.logical_seq, block_size);
    EXPECT_EQ(hashes.size(), 1u);

    const std::vector<llama_token> b0 = { 1, 2, 3, 4 };
    EXPECT_EQ(block_manager.lookup_block_by_hash(hashes[0], b0), group.block_table[0]);
    // Sanity check: release of non existent hash should not fail any asserts. It would be a no-op.
    block_manager.release_gpu_blocks({ group.block_table[1] });
}

TEST(test_register_prompt_blocks_short_prompt_no_op) {
    const uint32_t block_size   = 4;
    const uint32_t n_gpu_blocks = 2;
    const uint32_t n_cpu_blocks = 1;
    const float    watermark    = 0.0f;

    auto   kv            = llama_kv_cache_paged(64, 4, block_size, 2, 32, 8);
    auto & block_manager = kv.get_block_manager();
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    llama_sequence_group group{};
    group.request_id  = 0;
    group.n_prompt    = 3;                                     // less than one full block
    group.logical_seq = { 9, 9, 9 };
    group.block_table = block_manager.checkout_gpu_blocks(1);  // partial block held

    // Sanity checks: register and release no-ops
    kv.register_prompt_blocks(group);
    block_manager.release_gpu_blocks(group.block_table);
}

TEST(test_register_prompt_blocks_partial_block_not_findable) {
    const uint32_t block_size   = 4;
    const uint32_t n_gpu_blocks = 4;
    const uint32_t n_cpu_blocks = 2;
    const float    watermark    = 0.0f;

    auto   kv            = llama_kv_cache_paged(64, 4, block_size, 2, 32, 8);
    auto & block_manager = kv.get_block_manager();
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    llama_sequence_group group{};
    group.request_id  = 0;
    group.n_prompt    = 6;  // 1 full + 2 trailing toks
    group.logical_seq = { 1, 2, 3, 4, 5, 6 };
    group.block_table = block_manager.checkout_gpu_blocks(2);

    kv.register_prompt_blocks(group);

    // A future prompt that completes the trailing block must not see
    // a 2-block prefix match (because block 2 was never indexed).
    std::vector<llama_token> follow_up = { 1, 2, 3, 4, 5, 6, 7, 8 };  // 2 full blocks
    auto                     matches   = kv.find_prompt_prefix_matches(follow_up);
    EXPECT_EQ(matches.size(), 1u);                                    // we only match the first block
    EXPECT_EQ(matches[0], group.block_table[0]);
}

TEST(test_find_prompt_prefix_partial_match_stops_at_divergence) {
    const uint32_t block_size = 4;
    const float    watermark  = 0.0f;

    auto   kv            = llama_kv_cache_paged(64, 4, block_size, 2, 32, 8);
    auto & block_manager = kv.get_block_manager();
    block_manager.init(/*n_gpu=*/4, /*n_cpu=*/2, watermark);

    // Register prompt of 3 full blocks
    llama_sequence_group base{};
    base.request_id  = 0;
    base.n_prompt    = 12;
    base.logical_seq = { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 };
    base.block_table = block_manager.checkout_gpu_blocks(3);
    kv.register_prompt_blocks(base);

    // New prompt: matches block 0, diverges in block 1
    std::vector<llama_token> probe   = { 1, 1, 1, 1, 2, 2, 2, 8, 3, 3, 3, 3, 7 };
    auto                     matches = kv.find_prompt_prefix_matches(probe);
    EXPECT_EQ(matches.size(), 1u);  // exactly one (only bid 0)
    EXPECT_EQ(matches[0], base.block_table[0]);
}

TEST(test_find_prompt_prefix_full_match_drops_last_block) {
    // Invariant: a prompt that is exactly a multiple of block_size
    // and whose every block matches must not return all blocks.
    // We always reserve at least one block for prefill to avoid livelock.
    const uint32_t block_size = 4;
    const float    watermark  = 0.0f;

    auto   kv            = llama_kv_cache_paged(64, 4, block_size, 2, 32, 8);
    auto & block_manager = kv.get_block_manager();
    block_manager.init(/*n_gpu=*/4, /*n_cpu=*/2, watermark);

    std::vector<llama_token> toks = { 1, 2, 3, 4, 5, 6, 7, 8 };  // 2 full blocks, no partial

    llama_sequence_group base{};
    base.request_id  = 0;
    base.n_prompt    = toks.size();
    base.logical_seq = toks;
    base.block_table = block_manager.checkout_gpu_blocks(2);
    kv.register_prompt_blocks(base);

    auto matches = kv.find_prompt_prefix_matches(toks);
    EXPECT_EQ(matches.size(), 1u);
}

TEST(test_find_prompt_prefix_full_match_keeps_all_when_partial_trailing) {
    // Sanity counterpart to the above: when there IS a partial trailing block,
    // all full-block matches are returned because prefill always has >= 1 token.
    const uint32_t block_size = 4;
    const float    watermark  = 0.0f;

    auto   kv            = llama_kv_cache_paged(64, 4, block_size, 2, 32, 8);
    auto & block_manager = kv.get_block_manager();
    block_manager.init(/*n_gpu=*/4, /*n_cpu=*/2, watermark);

    std::vector<llama_token> regd = { 1, 2, 3, 4, 5, 6, 7, 8 };
    llama_sequence_group     base{};
    base.request_id  = 0;
    base.n_prompt    = regd.size();
    base.logical_seq = regd;
    base.block_table = block_manager.checkout_gpu_blocks(2);
    kv.register_prompt_blocks(base);

    std::vector<llama_token> probe   = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };  // 2 full + 2 trailing
    auto                     matches = kv.find_prompt_prefix_matches(probe);
    EXPECT_EQ(matches.size(), 2u);
}

TEST(test_find_prompt_prefix_no_refcount_change) {
    const uint32_t block_size = 4;
    const float    watermark  = 0.0f;

    auto   kv            = llama_kv_cache_paged(64, 4, block_size, 2, 32, 8);
    auto & block_manager = kv.get_block_manager();
    block_manager.init(/*n_gpu=*/4, /*n_cpu=*/2, watermark);

    llama_sequence_group base{};
    base.request_id  = 0;
    base.n_prompt    = 8;
    base.logical_seq = { 1, 2, 3, 4, 5, 6, 7, 8 };
    base.block_table = block_manager.checkout_gpu_blocks(2);
    kv.register_prompt_blocks(base);

    const uint32_t rc0_before = block_manager.get_ref_count(base.block_table[0]);
    const uint32_t rc1_before = block_manager.get_ref_count(base.block_table[1]);

    // Call find twice without claim — refcounts must be unchanged.
    std::vector<llama_token> probe = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    (void) kv.find_prompt_prefix_matches(probe);
    (void) kv.find_prompt_prefix_matches(probe);

    EXPECT_EQ(block_manager.get_ref_count(base.block_table[0]), rc0_before);
    EXPECT_EQ(block_manager.get_ref_count(base.block_table[1]), rc1_before);
}

TEST(test_claim_prompt_prefix_increments_and_appends) {
    const uint32_t block_size = 4;
    const float    watermark  = 0.0f;

    auto   kv            = llama_kv_cache_paged(64, 4, block_size, 2, 32, 8);
    auto & block_manager = kv.get_block_manager();
    block_manager.init(/*n_gpu=*/4, /*n_cpu=*/2, watermark);

    llama_sequence_group base{};
    base.request_id  = 0;
    base.n_prompt    = 8;
    base.logical_seq = { 1, 2, 3, 4, 5, 6, 7, 8 };
    base.block_table = block_manager.checkout_gpu_blocks(2);
    kv.register_prompt_blocks(base);

    std::vector<llama_token> probe   = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    auto                     matches = kv.find_prompt_prefix_matches(probe);
    EXPECT_EQ(matches.size(), 2u);

    kv.claim_prompt_prefix_matches(matches);

    EXPECT_EQ(block_manager.get_ref_count(base.block_table[0]), 2u);
    EXPECT_EQ(block_manager.get_ref_count(base.block_table[1]), 2u);
}

// CoW + prefix sharing interaction

TEST(test_finished_sharer_keeps_shared_blocks_alive) {
    const uint32_t block_size = 4;
    const float    watermark  = 0.0f;

    auto   kv            = llama_kv_cache_paged(64, 4, block_size, 2, 32, 8);
    auto & block_manager = kv.get_block_manager();
    block_manager.init(/*n_gpu=*/8, /*n_cpu=*/2, watermark);

    // A: 12-token prompt across 3 blocks
    llama_sequence_group a{};
    a.request_id  = 0;
    a.n_prompt    = 12;
    a.logical_seq = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    a.block_table = block_manager.checkout_gpu_blocks(3);
    kv.register_prompt_blocks(a);

    // B claims A's first two blocks via prefix match.
    std::vector<llama_token> probe   = { 1, 2, 3, 4, 5, 6, 7, 8, 99, 100 };
    auto                     matches = kv.find_prompt_prefix_matches(probe);
    EXPECT_EQ(matches.size(), 2u);

    llama_sequence_group b{};
    b.request_id  = 1;
    b.block_table = matches;
    kv.claim_prompt_prefix_matches(matches);
    EXPECT_EQ(block_manager.get_ref_count(matches[0]), 2u);
    EXPECT_EQ(block_manager.get_ref_count(matches[1]), 2u);

    // A finishes — release all 3 of A's blocks.
    kv.free_blocks(a);

    // B's claimed blocks survive with refcount 1, hashes preserved.
    EXPECT_EQ(block_manager.get_ref_count(matches[0]), 1u);
    EXPECT_EQ(block_manager.get_ref_count(matches[1]), 1u);

    auto matches_after = kv.find_prompt_prefix_matches(probe);
    EXPECT_EQ(matches_after.size(), 2u);
    EXPECT_EQ(matches_after[0], matches[0]);
    EXPECT_EQ(matches_after[1], matches[1]);

    // A's third block was returned to the pool — its hash entry should be erased,
    // so a probe extending past block 2 cannot match block 2.
    std::vector<llama_token> probe_all   = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
    auto                     matches_all = kv.find_prompt_prefix_matches(probe_all);
    EXPECT_EQ(matches_all.size(), 2u);

    block_manager.release_gpu_blocks(b.block_table);
}

TEST(test_last_sharer_release_clears_hash) {
    const uint32_t block_size = 4;
    const float    watermark  = 0.0f;

    auto   kv            = llama_kv_cache_paged(64, 4, block_size, 2, 32, 8);
    auto & block_manager = kv.get_block_manager();
    block_manager.init(/*n_gpu=*/8, /*n_cpu=*/2, watermark);

    llama_sequence_group a{};
    a.request_id  = 0;
    a.n_prompt    = 8;
    a.logical_seq = { 1, 2, 3, 4, 5, 6, 7, 8 };
    a.block_table = block_manager.checkout_gpu_blocks(2);
    kv.register_prompt_blocks(a);

    std::vector<llama_token> probe   = { 1, 2, 3, 4, 5, 6, 7, 8, 99, 100 };
    auto                     matches = kv.find_prompt_prefix_matches(probe);
    EXPECT_EQ(matches.size(), 2u);

    llama_sequence_group b{};
    b.request_id  = 1;
    b.block_table = matches;
    kv.claim_prompt_prefix_matches(matches);

    // Both sharers release
    kv.free_blocks(a);
    kv.free_blocks(b);

    // No live holders. Hashes must be gone.
    auto matches_after = kv.find_prompt_prefix_matches(probe);
    EXPECT_EQ(matches_after.size(), 0u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), 8u);
}

TEST(test_scheduler_prefix_share_through_step) {
    // End-to-end through the scheduler: A prefills + registers, then B is queued
    // with the same prompt and admitted with A's blocks reused via prefix sharing.
    const uint32_t block_size    = 4;
    const uint32_t n_prompt      = 16;  // 4 full blocks
    auto           fixture       = make_fixture(/*n_ctx=*/64, block_size, /*n_batch=*/64,
                                /*n_gpu_blocks=*/16, /*n_cpu_blocks=*/4);
    auto &         block_manager = fixture.kv->get_block_manager();

    auto group_a = make_group(/*request_id=*/0, n_prompt);
    EXPECT_TRUE(fixture.sched->queue_request(group_a));

    // Step 1: Prefil group A
    llama_batch batch  = {};
    auto        status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::OK);

    // Update A: registers all 4 of A's full prompt blocks
    std::vector<llama_token> new_tokens_1 = { 99 };
    std::vector<int8_t>      stop_flags_1 = { 0 };
    fixture.sched->update(batch, new_tokens_1, stop_flags_1.data());

    auto * a_ptr = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(a_ptr != nullptr);
    EXPECT_EQ(a_ptr->block_table.size(), 5u);

    // Queue B with the same prompt (logical_seq is all 1s by default)
    auto group_b = make_group(/*request_id=*/1, n_prompt);
    EXPECT_TRUE(fixture.sched->queue_request(group_b));

    // Step 2: A decode + B prefill (with shared prefix).
    status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::OK);

    auto * b_ptr = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(b_ptr != nullptr);

    // Prefix-sharing invariant (last matched block): so B claims A's first 3 blocks
    EXPECT_EQ(b_ptr->n_past, 12u);
    EXPECT_EQ(b_ptr->block_table.size(), 5u);  // 3 shared + 2 fresh
    EXPECT_EQ(b_ptr->block_table[0], a_ptr->block_table[0]);
    EXPECT_EQ(b_ptr->block_table[1], a_ptr->block_table[1]);
    EXPECT_EQ(b_ptr->block_table[2], a_ptr->block_table[2]);
    EXPECT_TRUE(b_ptr->block_table[3] != a_ptr->block_table[3]);
    EXPECT_TRUE(b_ptr->block_table[4] != a_ptr->block_table[4]);

    // ref_counts: shared blocks at 2, unshared blocks at 1.
    EXPECT_EQ(block_manager.get_ref_count(a_ptr->block_table[0]), 2u);
    EXPECT_EQ(block_manager.get_ref_count(a_ptr->block_table[1]), 2u);
    EXPECT_EQ(block_manager.get_ref_count(a_ptr->block_table[2]), 2u);
    EXPECT_EQ(block_manager.get_ref_count(a_ptr->block_table[3]), 1u);
    EXPECT_EQ(block_manager.get_ref_count(a_ptr->block_table[4]), 1u);
    EXPECT_EQ(block_manager.get_ref_count(b_ptr->block_table[3]), 1u);
    EXPECT_EQ(block_manager.get_ref_count(b_ptr->block_table[4]), 1u);
}

TEST(test_scheduler_prefix_share_admits_b_on_tight_budget) {
    // Verifies the budget benefit: B is admitted only because prefix sharing
    // reduced its batch token cost. With n_batch=17 and A consuming 1 decode
    // token, remaining budget is 16. Without prefix sharing B would need 17
    // tokens (full prompt + 1) and be rejected; with sharing it needs only
    // n_prompt - matched_tokens + 1 = 16 - 12 + 1 = 5 tokens and is admitted.
    const uint32_t block_size = 4;
    const uint32_t n_prompt   = 16;  // 4 full blocks
    auto           fixture    = make_fixture(/*n_ctx=*/64, block_size, /*n_batch=*/17,
                                /*n_gpu_blocks=*/16, /*n_cpu_blocks=*/4);

    auto group_a = make_group(/*request_id=*/0, n_prompt);
    EXPECT_TRUE(fixture.sched->queue_request(group_a));

    // Step 1: Prerfill group A (consumes the entire 17-token budget)
    llama_batch batch  = {};
    auto        status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::OK);

    std::vector<llama_token> new_tokens_1 = { 99 };
    std::vector<int8_t>      stop_flags_1 = { 0 };
    fixture.sched->update(batch, new_tokens_1, stop_flags_1.data());

    // Queue B with the same prompt — without the budget fix, B's tokens_needed
    // would be n_prompt + 1 = 17 > remaining 16 and B would be rejected.
    auto group_b = make_group(/*request_id=*/1, n_prompt);
    EXPECT_TRUE(fixture.sched->queue_request(group_b));

    // Step 2: A decodes (1 token), B prefills (4 unshared tokens). Total = 5 (still within budget)
    status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::OK);

    auto * b_ptr = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(b_ptr != nullptr);
    // B was admitted (running) and has its block_table populated.
    EXPECT_EQ(b_ptr->status, llama_sequence_group_status::RUNNING);

    const llama_paged_batch_info * info = fixture.sched->get_curr_batch_info();
    EXPECT_TRUE(info != nullptr);
    EXPECT_EQ(info->n_seq, 2);
    // batch_lens reflects what's actually written: 1 for A, 4 for B's unshared portion
    EXPECT_EQ(info->n_tokens, 5);
    EXPECT_EQ(info->batch_lens[0], 1);
    EXPECT_EQ(info->batch_lens[1], 4);
}

TEST(test_prefix_share_does_not_trigger_cow_on_decode) {
    const uint32_t block_size = 4;
    const float    watermark  = 0.0f;

    auto   kv            = llama_kv_cache_paged(64, 4, block_size, 2, 32, 8);
    auto & block_manager = kv.get_block_manager();
    block_manager.init(/*n_gpu=*/8, /*n_cpu=*/2, watermark);

    // A registers 2 prompt blocks
    llama_sequence_group a{};
    a.request_id  = 0;
    a.n_prompt    = 8;
    a.logical_seq = { 1, 2, 3, 4, 5, 6, 7, 8 };
    a.block_table = block_manager.checkout_gpu_blocks(2);
    kv.register_prompt_blocks(a);

    // B's prompt matches A's prompt exactly. Option D drops the last matched block.
    std::vector<llama_token> probe   = { 1, 2, 3, 4, 5, 6, 7, 8 };
    auto                     matches = kv.find_prompt_prefix_matches(probe);
    EXPECT_EQ(matches.size(), 1u);

    llama_sequence_group b{};
    b.request_id  = 1;
    b.n_prompt    = 8;
    b.logical_seq = probe;
    b.block_table = matches;
    kv.claim_prompt_prefix_matches(matches);
    EXPECT_EQ(block_manager.get_ref_count(matches[0]), 2u);

    // Reserve room for the rest of B's prefill + 1 generated token.
    bool ok = kv.allocate(b.n_prompt + 1, b);
    EXPECT_TRUE(ok);
    EXPECT_EQ(b.block_table.size(), 3u);

    // B's first write lands at position 4 → block_table[1], a freshly allocated,
    // unshared block. CoW must be a no-op.
    const uint32_t b_block_1_before = b.block_table[1];
    const size_t   free_before      = block_manager.n_free_gpu_blocks();

    bool cow_ok = kv.try_cow_shared_blocks(b, /*write_start_pos=*/4, /*n_write_tokens=*/1);
    EXPECT_TRUE(cow_ok);
    EXPECT_EQ(b.block_table[1], b_block_1_before);
    EXPECT_EQ(b.block_table[0], matches[0]);
    EXPECT_EQ(block_manager.get_ref_count(matches[0]), 2u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), free_before);
}

// Testing prefix-hashing

TEST(test_compute_prompt_block_hashes_empty_returns_empty) {
    const std::vector<llama_token> empty;
    EXPECT_TRUE(llama_kv_cache_paged::compute_prompt_block_hashes(empty, 16).empty());
}

TEST(test_compute_prompt_block_hashes_partial_block_excluded) {
    // 31 tokens, block_size 16 (1 full block and 15 trailing tokens dropped)
    std::vector<llama_token> toks(31, 7);
    auto                     hashes = llama_kv_cache_paged::compute_prompt_block_hashes(toks, 16);
    EXPECT_EQ(hashes.size(), 1u);
}

TEST(test_compute_prompt_block_hashes_full_block_count) {
    std::vector<llama_token> toks(48, 1);  // 3 full blocks
    auto                     hashes = llama_kv_cache_paged::compute_prompt_block_hashes(toks, 16);
    EXPECT_EQ(hashes.size(), 3u);
}

TEST(test_compute_prompt_block_hashes_deterministic) {
    std::vector<llama_token> toks     = { 1, 2, 3, 4, 5, 6, 7, 8 };
    auto                     hashes_a = llama_kv_cache_paged::compute_prompt_block_hashes(toks, 4);
    auto                     hashes_b = llama_kv_cache_paged::compute_prompt_block_hashes(toks, 4);
    EXPECT_EQ(hashes_a.size(), 2u);
    EXPECT_EQ(hashes_b.size(), 2u);
    EXPECT_EQ(hashes_a[0], hashes_b[0]);
    EXPECT_EQ(hashes_a[1], hashes_b[1]);
}

TEST(test_compute_prompt_block_hashes_chain_property) {
    // Two prompts with the same first block but different second block
    // The first hash will match, the second will differ
    std::vector<llama_token> toks_a   = { 1, 2, 3, 4, 10, 20, 30, 40 };
    std::vector<llama_token> toks_b   = { 1, 2, 3, 4, 10, 20, 30, 41 };
    auto                     hashes_a = llama_kv_cache_paged::compute_prompt_block_hashes(toks_a, 4);
    auto                     hashes_b = llama_kv_cache_paged::compute_prompt_block_hashes(toks_b, 4);
    EXPECT_EQ(hashes_a.size(), 2u);
    EXPECT_EQ(hashes_b.size(), 2u);
    EXPECT_EQ(hashes_a[0], hashes_b[0]);
    EXPECT_TRUE(hashes_a[1] != hashes_b[1]);
}

TEST(test_compute_prompt_block_hashes_position_sensitive) {
    // Same set of tokens but reordered should lead to different hash.
    // Testing XOR specifically.
    std::vector<llama_token> toks_a   = { 1, 2, 3, 4 };
    std::vector<llama_token> toks_b   = { 4, 3, 2, 1 };
    auto                     hashes_a = llama_kv_cache_paged::compute_prompt_block_hashes(toks_a, 4);
    auto                     hashes_b = llama_kv_cache_paged::compute_prompt_block_hashes(toks_b, 4);
    EXPECT_EQ(hashes_a.size(), 1u);
    EXPECT_EQ(hashes_b.size(), 1u);
    EXPECT_TRUE(hashes_a[0] != hashes_b[0]);
}

TEST(test_compute_prompt_block_hashes_zero_tokens_distinct_per_length) {
    // All-zero token streams of different (full-block) lengths must produce different
    // hahes for the full block. Testing specifically non zero seed.
    std::vector<llama_token> short_zeros(4, 0);
    std::vector<llama_token> long_zeros(8, 0);
    auto                     hashes_short = llama_kv_cache_paged::compute_prompt_block_hashes(short_zeros, 4);
    auto                     hashes_long  = llama_kv_cache_paged::compute_prompt_block_hashes(long_zeros, 4);
    EXPECT_EQ(hashes_short.size(), 1u);
    EXPECT_EQ(hashes_long.size(), 2u);
    EXPECT_TRUE(hashes_short[0] != hashes_long[1]);
}

int main(int /*argc*/, char ** /*argv*/) {
    fprintf(stderr, "1) test-paged-kv: block_manager\n");
    RUN(test_block_manager_leak_simple);
    RUN(test_block_manager_leak_repeated);
    RUN(test_block_manager_checkout_too_many);
    RUN(test_block_manager_watermark);
    RUN(test_block_manager_watermark_zero);
    RUN(test_block_manager_gpu_cpu_disjoint);
    RUN(test_block_manager_shared_gpu_block);
    RUN(test_block_manager_register_lookup_roundtrip);
    RUN(test_block_manager_lookup_collision_mismatched_tokens);
    RUN(test_block_manager_lookup_miss);
    RUN(test_block_manager_release_clears_hash);
    RUN(test_block_manager_shared_block_keeps_hash_until_zero);

    fprintf(stderr, "\n\n2) test-paged-kv: llama_kv_cache_paged seq_pos\n");
    RUN(test_seq_pos_default_unknown);
    RUN(test_seq_pos_set_and_get);
    RUN(test_seq_pos_overwrite);
    RUN(test_seq_pos_seq_rm_removes);
    RUN(test_seq_pos_clear_removes_all);
    RUN(test_seq_cp_shares_blocks);

    fprintf(stderr, "\n\n3) test-paged-kv: llama_kv_cache_paged free_blocks\n");
    RUN(test_free_blocks_releases_to_pool);

    fprintf(stderr, "\n\n4) test-paged-kv: llama_kv_cache_paged scheduler\n");
    RUN(test_scheduler_no_deadlock_on_empty);
    RUN(test_scheduler_deadlock_oversize_waiting_request);
    RUN(test_scheduler_rejects_oversized_prompt);
    RUN(test_cow_writes_to_shared_block);
    RUN(test_cow_no_op_for_unshared_block);
    RUN(test_cow_fails_under_gpu_pressure);

    fprintf(stderr, "\n\n5) test-paged-kv: llama_kv_cache_paged auto prompt hashing\n");
    RUN(test_register_prompt_blocks_full_blocks);
    RUN(test_register_prompt_blocks_partial_block_ignored);
    RUN(test_register_prompt_blocks_short_prompt_no_op);
    RUN(test_register_prompt_blocks_partial_block_not_findable);

    RUN(test_find_prompt_prefix_no_refcount_change);
    RUN(test_find_prompt_prefix_full_match_drops_last_block);
    RUN(test_find_prompt_prefix_partial_match_stops_at_divergence);
    RUN(test_find_prompt_prefix_full_match_keeps_all_when_partial_trailing);
    RUN(test_claim_prompt_prefix_increments_and_appends);

    RUN(test_finished_sharer_keeps_shared_blocks_alive);
    RUN(test_last_sharer_release_clears_hash);
    RUN(test_scheduler_prefix_share_through_step);
    RUN(test_scheduler_prefix_share_admits_b_on_tight_budget);
    RUN(test_prefix_share_does_not_trigger_cow_on_decode);

    fprintf(stderr, "\n\n6) test-paged-kv: prefix hashing\n");
    RUN(test_compute_prompt_block_hashes_empty_returns_empty);
    RUN(test_compute_prompt_block_hashes_partial_block_excluded);
    RUN(test_compute_prompt_block_hashes_full_block_count);
    RUN(test_compute_prompt_block_hashes_deterministic);
    RUN(test_compute_prompt_block_hashes_position_sensitive);
    RUN(test_compute_prompt_block_hashes_zero_tokens_distinct_per_length);

    fprintf(stderr, "\n\ntest-paged-kv: ALL PASSED\n");
    return 0;
}
