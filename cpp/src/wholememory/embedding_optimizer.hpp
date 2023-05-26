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

#include <wholememory/embedding.h>

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "embedding_cache.hpp"

#ifdef __cplusplus
extern "C" {
#endif

struct wholememory_embedding_optimizer_ {
  wholememory_optimizer_type_t optimizer_type;
};

#ifdef __cplusplus
}
#endif

namespace wholememory {

class embedding_optimizer_impl_base;

using optimizer_parameter_setter_fn_t = std::function<wholememory_error_code_t(const void*)>;

class optimizer_state_t {
 public:
  optimizer_state_t()  = default;
  ~optimizer_state_t() = default;
  // Per element optimizer states are cachable, like momentums.
  // They are packed into same
  struct cachable_state {
    // name of this state
    std::string name;
    int start_dim;
    int dim;
    wholememory_tensor_t global_raw_state_tensor = nullptr;
  };
  wholememory_embedding_t cachable_state_embedding = nullptr;

  // wholememory_tensor_t global_cachable_raw_padded_tensor = nullptr;
  wholememory_tensor_t global_cachable_raw_user_tensor = nullptr;
  wholememory_tensor_t local_cachable_wm_tensor        = nullptr;

  // wholememory_tensor_t global_cacheline_tag_wm_tensor    = nullptr;
  // wholememory_tensor_t global_cacheline_data_wm_tensor   = nullptr;
  // wholememory_tensor_t local_cacheline_tag_wm_tensor     = nullptr;
  // wholememory_tensor_t local_cacheline_data_wm_tensor    = nullptr;
  //  per embedding optimizers are uncachable, like betat1 and batat2 for momentums.
  struct uncachable_state {
    std::string name;
    int dim;
    wholememory_dtype_t dtype;
    wholememory_tensor_t global_raw_padded_tensor = nullptr;
    wholememory_tensor_t global_raw_sub_tensor    = nullptr;
    wholememory_tensor_t local_tensor             = nullptr;
  };
  int64_t local_start_index                     = -1;
  device_cache_for_host* device_cache_for_host_ = nullptr;
  std::vector<cachable_state> cachable_states;
  std::vector<uncachable_state> uncachable_states;
};

class embedding_optimizer_impl_base : public wholememory_embedding_optimizer_ {
 public:
  embedding_optimizer_impl_base();
  virtual ~embedding_optimizer_impl_base() = default;
  virtual wholememory_error_code_t set_parameter(const char* parameter_name, void* value) noexcept;
  /**
   * Apply gradients.
   * As trainable Embedding use READWRITE cache, Cache communicator is the same as Embedding
   * communicator. Gradients will be partitioned and each rank is only responsible for its own
   * partition.
   *
   * @param indices : bucketed indices that belongs to current rank.
   * @param grads : bucketed gradients that belongs to current rank.
   * @param local_embedding : local embedding of current rank.
   * @param optimizer_state : pointer to optimizer state.
   * @param lr : learning rate
   * @param stream : cudaStream_t to use
   * @return : wholememory_error_code_t
   */
  virtual wholememory_error_code_t step(wholememory_tensor_t indices,
                                        wholememory_tensor_t grads,
                                        wholememory_tensor_t local_embedding,
                                        optimizer_state_t* optimizer_state,
                                        float lr,
                                        cudaStream_t stream) noexcept = 0;

  virtual void create_optimizer_states(optimizer_state_t* optimizer_state,
                                       int embedding_dim) noexcept
  {
  }

  virtual wholememory_error_code_t init_optimizer_states(
    optimizer_state_t* optimizer_state) noexcept
  {
    return WHOLEMEMORY_SUCCESS;
  }
  [[nodiscard]] const char* const* get_optimizer_state_names() const noexcept
  {
    return state_names_.data();
  }
  virtual wholememory_tensor_t get_optimizer_state(optimizer_state_t* optimizer_state,
                                                   const char* state_name);

 protected:
  static optimizer_parameter_setter_fn_t get_float_setter(float* target_ptr);
  static void zero_local_state_tensor(wholememory_tensor_t local_state_tensor);
  static void set_float_local_state_tensor(wholememory_tensor_t local_state_tensor, float value);

  std::map<std::string, optimizer_parameter_setter_fn_t> setter_fns_;
  const char* name_ = nullptr;

  std::vector<const char*> state_names_ = {nullptr};
};

wholememory_error_code_t create_embedding_optimizer(
  wholememory_embedding_optimizer_t* optimizer,
  wholememory_optimizer_type_t optimizer_type) noexcept;

wholememory_error_code_t optimizer_set_parameter(wholememory_embedding_optimizer_t optimizer,
                                                 const char* parameter_name,
                                                 void* value) noexcept;

void destroy_embedding_optimizer(wholememory_embedding_optimizer_t optimizer) noexcept;

}  // namespace wholememory
