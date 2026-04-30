#pragma once

#include "llama.h"  // llama_token

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

class llama_block_manager {
    struct PhysicalBlock {
        uint32_t id        = 0;
        uint32_t ref_count = 0;
        bool     is_gpu    = false;
    };

    using PhysicalBlockPool = std::vector<PhysicalBlock>;
    using PhysicalBlockIds  = std::vector<uint32_t>;
    using TokenList         = std::vector<llama_token>;

    struct PhysicalBlockMeta {
        uint64_t  hash;
        TokenList tokens;  // for collision verification
    };

    PhysicalBlockPool gpu_registry;
    PhysicalBlockPool cpu_registry;

    PhysicalBlockIds free_gpu_ids;  // [0 to total_num_gpu_blocks - 1]
    PhysicalBlockIds free_cpu_ids;  // [total_num_gpu_blocks, total_num_gpu_blocks + total_num_cpu_blocks]

    std::unordered_map<uint64_t, uint32_t>          hash_to_block;
    std::unordered_map<uint32_t, PhysicalBlockMeta> block_to_meta;

    uint32_t watermark_gpu_safety_num_blocks;
    uint32_t watermark_cpu_safety_num_blocks;

    uint32_t total_num_gpu_blocks;
    uint32_t total_num_cpu_blocks;

  public:
    static constexpr uint32_t INVALID_BLOCK_ID = UINT32_MAX;

    void init(uint32_t n_gpu, uint32_t n_cpu, float watermark);

    size_t n_free_gpu_blocks() const;
    size_t n_free_cpu_blocks() const;

    bool has_free_gpu_blocks(uint32_t num_requested_blocks) const;
    bool has_free_cpu_blocks(uint32_t num_requested_blocks) const;

    PhysicalBlockIds checkout_gpu_blocks(uint32_t num_blocks);
    PhysicalBlockIds checkout_cpu_blocks(uint32_t num_blocks);

    void release_gpu_blocks(const PhysicalBlockIds & freed_blocks);
    void release_cpu_blocks(const PhysicalBlockIds & freed_blocks);

    void     increment_ref(uint32_t block_id);
    uint32_t get_ref_count(uint32_t block_id) const;  // only GPU

    bool is_gpu(uint32_t block) const;

    // Registers a (hash, token) to block_id mapping.
    // Idempotent: if hash slot is alreadsy taken, no-op (first write wins)
    void register_block_hash(uint32_t block_id, uint64_t hash, TokenList tokens);

    // Returns the block_id whose registered tokens equal 'tokens' (i.e. real
    // hit, not just a hash collision). Returns INVALID_BLOCK_ID on miss
    // or collision with mismatched tokens.
    // Note: Caller is responsible for increment_ref on hit.
    uint32_t lookup_block_by_hash(uint64_t hash, const TokenList & tokens) const;
};
