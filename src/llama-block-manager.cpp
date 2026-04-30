#include "llama-block-manager.h"

#include "llama-impl.h"

#include <cmath>

void llama_block_manager::init(uint32_t n_gpu, uint32_t n_cpu, float watermark) {
    LLAMA_LOG_INFO("%s: Block manager initialized: n_free_gpu_blocks=%d, n_free_cpu_blocks=%d\n", __func__, n_gpu,
                   n_cpu);
    total_num_gpu_blocks = n_gpu;
    total_num_cpu_blocks = n_cpu;

    watermark_gpu_safety_num_blocks = std::ceil(total_num_gpu_blocks * watermark);
    watermark_cpu_safety_num_blocks = std::ceil(total_num_cpu_blocks * watermark);

    gpu_registry.resize(n_gpu);
    for (uint32_t i = 0; i < n_gpu; ++i) {
        gpu_registry[i].id     = i;
        gpu_registry[i].is_gpu = true;
        free_gpu_ids.push_back(i);
    }

    cpu_registry.resize(n_cpu);
    for (uint32_t i = 0; i < n_cpu; ++i) {
        cpu_registry[i].id     = i + total_num_gpu_blocks;
        cpu_registry[i].is_gpu = false;
        free_cpu_ids.push_back(i + total_num_gpu_blocks);
    }
}

size_t llama_block_manager::n_free_gpu_blocks() const {
    return free_gpu_ids.size();
}

size_t llama_block_manager::n_free_cpu_blocks() const {
    return free_cpu_ids.size();
}

bool llama_block_manager::has_free_gpu_blocks(uint32_t num_requested_blocks) const {
    size_t curr_free_gpus = free_gpu_ids.size();
    if (curr_free_gpus < watermark_gpu_safety_num_blocks) {
        return false;
    }
    return num_requested_blocks <= (curr_free_gpus - watermark_gpu_safety_num_blocks);
}

bool llama_block_manager::has_free_cpu_blocks(uint32_t num_requested_blocks) const {
    size_t curr_free_cpus = free_cpu_ids.size();
    if (curr_free_cpus < watermark_cpu_safety_num_blocks) {
        return false;
    }
    return num_requested_blocks <= (curr_free_cpus - watermark_cpu_safety_num_blocks);
}

llama_block_manager::PhysicalBlockIds llama_block_manager::checkout_gpu_blocks(uint32_t num_blocks) {
    PhysicalBlockIds new_ids = {};
    if (num_blocks > free_gpu_ids.size()) {
        return new_ids;
    }

    new_ids.insert(new_ids.end(), std::make_move_iterator(free_gpu_ids.end() - num_blocks),
                   std::make_move_iterator(free_gpu_ids.end()));
    free_gpu_ids.erase(free_gpu_ids.end() - num_blocks, free_gpu_ids.end());

    for (const uint32_t & id : new_ids) {
        gpu_registry[id].ref_count += 1;
    }
    return new_ids;
}

llama_block_manager::PhysicalBlockIds llama_block_manager::checkout_cpu_blocks(uint32_t num_blocks) {
    PhysicalBlockIds new_ids = {};
    if (num_blocks > free_cpu_ids.size()) {
        return new_ids;
    }

    new_ids.insert(new_ids.end(), std::make_move_iterator(free_cpu_ids.end() - num_blocks),
                   std::make_move_iterator(free_cpu_ids.end()));
    free_cpu_ids.erase(free_cpu_ids.end() - num_blocks, free_cpu_ids.end());

    for (const uint32_t & id : new_ids) {
        cpu_registry[id - total_num_gpu_blocks].ref_count += 1;
    }
    return new_ids;
}

void llama_block_manager::release_gpu_blocks(const PhysicalBlockIds & freed_blocks_ids) {
    for (const uint32_t & id : freed_blocks_ids) {
        gpu_registry[id].ref_count -= 1;
        if (gpu_registry[id].ref_count <= 0) {
            gpu_registry[id].ref_count = 0;

            // Drop hash registration before returning the block to the free pool
            auto meta_it = block_to_meta.find(id);
            if (meta_it != block_to_meta.end()) {
                hash_to_block.erase(meta_it->second.hash);
                block_to_meta.erase(meta_it);
            }
            free_gpu_ids.push_back(id);
        }
    }
}

void llama_block_manager::release_cpu_blocks(const PhysicalBlockIds & freed_blocks_ids) {
    for (const uint32_t & id : freed_blocks_ids) {
        cpu_registry[id - total_num_gpu_blocks].ref_count -= 1;
        if (cpu_registry[id - total_num_gpu_blocks].ref_count <= 0) {
            cpu_registry[id - total_num_gpu_blocks].ref_count = 0;
            free_cpu_ids.push_back(id);
        }
    }
}

void llama_block_manager::increment_ref(uint32_t block_id) {
    // Only increment checked-out GPU blocks
    GGML_ASSERT(!gpu_registry.empty() && "gpu_registry is empty.");
    GGML_ASSERT(block_id < total_num_gpu_blocks && "increment ref only supported for GPU blocks.");
    GGML_ASSERT(gpu_registry[block_id].ref_count > 0 && "can only increment checked-out GPU blocks.");
    gpu_registry[block_id].ref_count += 1;
}

uint32_t llama_block_manager::get_ref_count(uint32_t block_id) const {
    GGML_ASSERT(!gpu_registry.empty() && "gpu_registry is empty.");
    GGML_ASSERT(block_id < total_num_gpu_blocks && "get ref count only supported for GPU blocks.");
    return gpu_registry[block_id].ref_count;
}

bool llama_block_manager::is_gpu(uint32_t block_id) const {
    return block_id < total_num_gpu_blocks;
}

void llama_block_manager::register_block_hash(uint32_t block_id, uint64_t hash, TokenList tokens) {
    GGML_ASSERT(block_id < total_num_gpu_blocks && "hash registration only supported for GPU blocks.");
    GGML_ASSERT(gpu_registry[block_id].ref_count > 0 && "cannot register a freed block.");

    if (hash_to_block.count(hash)) {
        // Hash already registered (either the same block or a collision).
        // First write wins, so no-op.
        return;
    }

    hash_to_block[hash]     = block_id;
    block_to_meta[block_id] = { hash, std::move(tokens) };
}

uint32_t llama_block_manager::lookup_block_by_hash(uint64_t hash, const TokenList & tokens) const {
    auto it = hash_to_block.find(hash);
    if (it == hash_to_block.end()) {
        return INVALID_BLOCK_ID;
    }
    const uint32_t bid     = it->second;
    auto           meta_it = block_to_meta.find(bid);
    GGML_ASSERT(meta_it != block_to_meta.end() && "hash_to_block & block_to_meta out of sync.");

    if (meta_it->second.tokens != tokens) {
        return INVALID_BLOCK_ID;  // hash collision with different tokens
    }
    return bid;
}
