/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_runtime_api.h>

#include <wholememory/embedding.h>
#include <wholememory/wholememory_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

struct wholememory_embedding_cache_policy_ {
  wholememory_comm_t cache_comm = nullptr;  // nullptr means only use local GPU
  wholememory_memory_type_t cache_memory_type;
  wholememory_memory_location_t cache_memory_location;
  wholememory_access_type_t access_type;
  float cache_ratio = 0.2F;
};

#ifdef __cplusplus
}
#endif

namespace wholememory {

class embedding_cache_local_data {
 public:
  embedding_cache_local_data() = default;
  ~embedding_cache_local_data();

  wholememory_tensor_t cache_line_tag_       = nullptr;
  wholememory_tensor_t cache_line_lfu_count_ = nullptr;
  wholememory_tensor_t cache_line_data_      = nullptr;
  wholememory_tensor_t access_count_         = nullptr;
};

class embedding_cache_base {
 public:
  explicit embedding_cache_base(wholememory_embedding_cache_policy_t cache_policy);
  embedding_cache_base()                            = delete;
  embedding_cache_base(const embedding_cache_base&) = delete;
  virtual ~embedding_cache_base();

  embedding_cache_local_data* get_cache_local_data() { return &local_cache_; }
  [[nodiscard]] int get_cache_set_coverage() const { return cache_set_coverage_; }

  virtual wholememory_error_code_t get_embedding_requirement(
    wholememory_tensor_description_t* padded_desc,
    wholememory_matrix_description_t data_desc,
    wholememory_comm_t comm,
    wholememory_memory_type_t memory_type,
    wholememory_memory_location_t memory_location) noexcept = 0;

  wholememory_error_code_t allocate(wholememory_tensor_t raw_data_tensor) noexcept;

  virtual wholememory_error_code_t writeback_all_cache(cudaStream_t stream) noexcept;
  virtual wholememory_error_code_t drop_all_cache(cudaStream_t stream) noexcept;
  // wholememory_error_code_t refill_all_cache(cudaStream_t stream) noexcept;

  static constexpr int64_t kEmbeddingAlignmentInBytes = 16;
  static constexpr int kCacheSetSize                  = 32;
  // Tag format:
  // 1 bit Valid, 1 bit Modified, 14 bit indice.
  static constexpr int kCacheSetCoverageBits    = 14;  // 2 bits left for modified and valid state
  static constexpr uint16_t kvalidCacheTagValue = 1U << (kCacheSetCoverageBits + 1);
  static constexpr int kMaxCacheSetCoverage     = 1 << kCacheSetCoverageBits;
  // Counter format:
  // 14 bit scaled counter, 2 bit per thread (64 bit per set) set scaling info.
  static constexpr int kScaledCounterBits =
    14;  // 2 bits (64 bits in set) left for scale and reserved

  // cache related tensor
  wholememory_tensor_t cache_line_tag_wm_tensor_       = nullptr;
  wholememory_tensor_t cache_line_lfu_count_wm_tensor_ = nullptr;
  wholememory_tensor_t cache_line_data_wm_tensor_      = nullptr;
  wholememory_tensor_t access_count_wm_tensor_         = nullptr;

 protected:
  void pad_last_dim(wholememory_matrix_description_t data_desc) noexcept;
  wholememory_error_code_t compute_cache_set_coverage() noexcept;
  wholememory_error_code_t check_raw_tensor(wholememory_tensor_t raw_data_tensor) noexcept;

  wholememory_matrix_description_t padded_matrix_description_;
  wholememory_matrix_description_t matrix_description_;

  wholememory_tensor_t padded_raw_tensor_            = nullptr;  // just a reference, not owned
  wholememory_comm_t raw_comm_                       = nullptr;
  wholememory_memory_type_t raw_memory_type_         = WHOLEMEMORY_MT_NONE;
  wholememory_memory_location_t raw_memory_location_ = WHOLEMEMORY_ML_NONE;

  wholememory_embedding_cache_policy_t cache_policy_ = nullptr;

  int cache_set_coverage_                   = kCacheSetSize;
  int64_t padded_embedding_count_for_cache_ = 0;

  embedding_cache_local_data local_cache_;
};

class device_cache_for_host : public embedding_cache_base {
 public:
  device_cache_for_host(wholememory_embedding_cache_policy_t cache_policy);
  device_cache_for_host()                             = delete;
  device_cache_for_host(const device_cache_for_host&) = delete;
  ~device_cache_for_host();
  wholememory_error_code_t get_embedding_requirement(
    wholememory_tensor_description_t* padded_desc,
    wholememory_matrix_description_t data_desc,
    wholememory_comm_t comm,
    wholememory_memory_type_t memory_type,
    wholememory_memory_location_t memory_location) noexcept override;
  wholememory_error_code_t writeback_all_cache(cudaStream_t stream) noexcept override;
  wholememory_error_code_t drop_all_cache(cudaStream_t stream) noexcept override;
};

class local_cache_for_global : public embedding_cache_base {
 public:
  local_cache_for_global(wholememory_embedding_cache_policy_t cache_policy);
  local_cache_for_global()                              = delete;
  local_cache_for_global(const local_cache_for_global&) = delete;
  ~local_cache_for_global();
  wholememory_error_code_t get_embedding_requirement(
    wholememory_tensor_description_t* padded_desc,
    wholememory_matrix_description_t data_desc,
    wholememory_comm_t comm,
    wholememory_memory_type_t memory_type,
    wholememory_memory_location_t memory_location) noexcept override;
  wholememory_error_code_t drop_all_cache(cudaStream_t stream) noexcept override;
};

}  // namespace wholememory
