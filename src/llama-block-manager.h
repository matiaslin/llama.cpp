#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class llama_block_manager {
    struct PhysicalBlock {
        uint32_t id        = 0;
        uint32_t ref_count = 0;
        bool     is_gpu    = false;
    };

    using PhysicalBlockPool = std::vector<PhysicalBlock>;
    using PhysicalBlockIds  = std::vector<uint32_t>;

    PhysicalBlockPool gpu_registry;
    PhysicalBlockPool cpu_registry;

    PhysicalBlockIds free_gpu_ids;  // [0 to total_num_gpu_blocks - 1]
    PhysicalBlockIds free_cpu_ids;  // [total_num_gpu_blocks, total_num_gpu_blocks + total_num_cpu_blocks]

    uint32_t watermark_gpu_safety_num_blocks;
    uint32_t watermark_cpu_safety_num_blocks;

    uint32_t total_num_gpu_blocks;
    uint32_t total_num_cpu_blocks;

  public:
    void init(uint32_t n_gpu, uint32_t n_cpu, float watermark);

    size_t n_free_gpu_blocks() const;
    size_t n_free_cpu_blocks() const;

    bool has_free_gpu_blocks(uint32_t num_requested_blocks) const;
    bool has_free_cpu_blocks(uint32_t num_requested_blocks) const;

    PhysicalBlockIds checkout_gpu_blocks(uint32_t num_blocks);
    PhysicalBlockIds checkout_cpu_blocks(uint32_t num_blocks);

    void release_gpu_blocks(const PhysicalBlockIds & freed_blocks);
    void release_cpu_blocks(const PhysicalBlockIds & freed_blocks);

    bool is_gpu(uint32_t block) const;
};
