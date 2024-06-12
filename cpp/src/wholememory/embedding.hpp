/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <wholememory/embedding.h>
#include <wholememory/wholememory_tensor.h>

#include "embedding_optimizer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

struct wholememory_embedding_ {
  wholememory_tensor_t allocated_embedding          = nullptr;
  wholememory_tensor_t user_embedding               = nullptr;  // subtensor of allocated_embedding
  wholememory_embedding_cache_policy_t cache_policy = nullptr;
  wholememory_embedding_optimizer_t optimizer       = nullptr;
};

#ifdef __cplusplus
}
#endif

namespace wholememory {

class embedding_base : public wholememory_embedding_ {
 public:
  embedding_base()          = default;
  virtual ~embedding_base() = default;
  wholememory_error_code_t allocate(wholememory_matrix_description_t* embedding_description,
                                    wholememory_comm_t comm,
                                    wholememory_memory_type_t memory_type,
                                    wholememory_memory_location_t memory_location,
                                    wholememory_embedding_cache_policy_t policy) noexcept;
  void deallocate() noexcept;
  virtual wholememory_error_code_t gather(wholememory_tensor_t indices,
                                          wholememory_tensor_t output,
                                          bool adjust_cache,
                                          wholememory_env_func_t* p_env_fns,
                                          cudaStream_t stream) noexcept = 0;

  wholememory_error_code_t gather_gradient_apply(wholememory_tensor_t indices,
                                                 wholememory_tensor_t grads,
                                                 bool adjust_cache,
                                                 float lr,
                                                 wholememory_env_func_t* p_env_fns,
                                                 cudaStream_t stream);

  wholememory_error_code_t set_optimizer(wholememory_embedding_optimizer_t opt);

  [[nodiscard]] const char* const* get_optimizer_state_names() const noexcept
  {
    if (optimizer_impl_base_ != nullptr) {
      return optimizer_impl_base_->get_optimizer_state_names();
    }
    return nullptr;
  }
  virtual wholememory_tensor_t get_optimizer_state(const char* state_name) const noexcept
  {
    if (optimizer_impl_base_ != nullptr) {
      return optimizer_impl_base_->get_optimizer_state(optimizer_state_.get(), state_name);
    }
    return nullptr;
  }
  virtual wholememory_error_code_t writeback_embedding_cache(cudaStream_t stream) const noexcept;
  virtual wholememory_error_code_t writeback_all_caches(cudaStream_t stream) const noexcept;
  virtual wholememory_error_code_t drop_embedding_cache(cudaStream_t stream) const noexcept;
  virtual wholememory_error_code_t drop_all_caches(cudaStream_t stream) const noexcept;

  wholememory::embedding_cache_base* get_cache_ptr() const { return cache_ptr_; }
  wholememory_error_code_t set_shard_method(
    wholememory_matrix_description_t* embedding_matrix_description,
    int embedding_world_size,
    int round_robin_size) noexcept;
  wholememory_error_code_t set_gather_sms(int sms) noexcept;
  int get_round_robin_size() noexcept;

 protected:
  virtual wholememory_error_code_t init_optimizer_states() noexcept
  {
    if (optimizer_impl_base_ != nullptr) {
      WHOLEMEMORY_RETURN_ON_FAIL(
        optimizer_impl_base_->init_optimizer_states(optimizer_state_.get()));
      WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_barrier(raw_embedding_comm_));
      return WHOLEMEMORY_SUCCESS;
    }
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  wholememory_error_code_t create_optimizer_states() noexcept;
  wholememory_error_code_t destroy_optimizer_states() noexcept;

  int gather_sms_;
  int round_robin_size_;
  wholememory_dtype_t embedding_dtype_                             = WHOLEMEMORY_DT_UNKNOWN;
  wholememory_comm_t raw_embedding_comm_                           = nullptr;
  wholememory::embedding_cache_base* cache_ptr_                    = nullptr;
  wholememory::embedding_optimizer_impl_base* optimizer_impl_base_ = nullptr;
  std::unique_ptr<wholememory::optimizer_state_t> optimizer_state_ = nullptr;
};

}  // namespace wholememory
