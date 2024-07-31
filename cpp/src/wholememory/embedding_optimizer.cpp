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
#include "embedding_optimizer.hpp"

#include <cstring>

#include "cuda_macros.hpp"
#include "logger.hpp"
#include "wholememory/embedding.hpp"
#include "wholememory_ops/functions/embedding_optimizer_func.h"

namespace wholememory {

embedding_optimizer_impl_base::embedding_optimizer_impl_base() = default;

wholememory_error_code_t float_setter_fn(float* target_ptr, const void* data)
{
  const auto* float_data = static_cast<const float*>(data);
  *target_ptr            = *float_data;
  return WHOLEMEMORY_SUCCESS;
}

optimizer_parameter_setter_fn_t embedding_optimizer_impl_base::get_float_setter(float* target)
{
  return std::bind(float_setter_fn, target, std::placeholders::_1);
}

void embedding_optimizer_impl_base::zero_local_state_tensor(wholememory_tensor_t local_state_tensor)
{
  void* local_ptr        = wholememory_tensor_get_data_pointer(local_state_tensor);
  auto* local_state_desc = wholememory_tensor_get_tensor_description(local_state_tensor);
  WHOLEMEMORY_CHECK_NOTHROW(local_state_desc->storage_offset == 0);
  size_t total_elt_count = wholememory_get_memory_element_count_from_tensor(local_state_desc);
  size_t elt_size        = wholememory_dtype_get_element_size(local_state_desc->dtype);
  size_t total_size      = total_elt_count * elt_size;
  WM_CUDA_CHECK_NO_THROW(cudaMemset(local_ptr, 0, total_size));
  WM_CUDA_CHECK_NO_THROW(cudaDeviceSynchronize());
}

void embedding_optimizer_impl_base::set_float_local_state_tensor(
  wholememory_tensor_t local_state_tensor, float value)
{
  void* local_ptr        = wholememory_tensor_get_data_pointer(local_state_tensor);
  auto* local_state_desc = wholememory_tensor_get_tensor_description(local_state_tensor);
  WHOLEMEMORY_CHECK_NOTHROW(local_state_desc->storage_offset == 0);
  size_t total_elt_count = wholememory_get_memory_element_count_from_tensor(local_state_desc);
  wholememory_ops::set_memory_to_float_value(
    static_cast<float*>(local_ptr), value, total_elt_count, nullptr);
  WM_CUDA_CHECK_NO_THROW(cudaDeviceSynchronize());
}

wholememory_error_code_t embedding_optimizer_impl_base::set_parameter(const char* parameter_name,
                                                                      void* value) noexcept
{
  std::string const parameter_name_str = parameter_name;
  auto it                              = setter_fns_.find(parameter_name_str);
  if (it == setter_fns_.end()) {
    WHOLEMEMORY_ERROR("parameter name %s is not valid for optimizer %s", parameter_name, name_);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  return it->second(value);
}

wholememory_tensor_t embedding_optimizer_impl_base::get_optimizer_state(
  optimizer_state_t* optimizer_state, const char* state_name)
{
  WHOLEMEMORY_CHECK_NOTHROW(optimizer_state != nullptr);
  WHOLEMEMORY_CHECK_NOTHROW(state_names_.size() == optimizer_state->cachable_states.size() +
                                                     optimizer_state->uncachable_states.size() + 1);
  for (size_t i = 0; i < optimizer_state->cachable_states.size(); i++) {
    if (strcmp(state_name, optimizer_state->cachable_states[i].name.c_str()) == 0) {
      WHOLEMEMORY_CHECK_NOTHROW(strcmp(state_name, state_names_[i]) == 0);
      return optimizer_state->cachable_states[i].global_raw_state_tensor;
    }
  }
  size_t cachable_state_count = optimizer_state->cachable_states.size();
  for (size_t i = 0; i < optimizer_state->uncachable_states.size(); i++) {
    if (strcmp(state_name, optimizer_state->uncachable_states[i].name.c_str()) == 0) {
      WHOLEMEMORY_CHECK_NOTHROW(strcmp(state_name, state_names_[i + cachable_state_count]) == 0);
      return optimizer_state->uncachable_states[i].global_raw_sub_tensor;
    }
  }
  WHOLEMEMORY_FAIL_NOTHROW("optimizer state name %s not found for %s", state_name, name_);
  return nullptr;
}

class SGDEmbeddingOptimizer : public embedding_optimizer_impl_base {
 public:
  SGDEmbeddingOptimizer();
  wholememory_error_code_t step(wholememory_tensor_t indices,
                                wholememory_tensor_t grads,
                                wholememory_tensor_t local_embedding,
                                optimizer_state_t* optimizer_state,
                                float lr,
                                cudaStream_t stream) noexcept override;

 protected:
  float weight_decay = 0.0F;
};

SGDEmbeddingOptimizer::SGDEmbeddingOptimizer()
{
  name_ = "SGD";
  setter_fns_.emplace(std::pair<std::string, optimizer_parameter_setter_fn_t>(
    "weight_decay", get_float_setter(&weight_decay)));
  state_names_ = {nullptr};
}

wholememory_error_code_t SGDEmbeddingOptimizer::step(wholememory_tensor_t indices,
                                                     wholememory_tensor_t grads,
                                                     wholememory_tensor_t local_embedding,
                                                     optimizer_state_t* optimizer_state,
                                                     float lr,
                                                     cudaStream_t stream) noexcept
{
  WHOLEMEMORY_CHECK_NOTHROW(grads != nullptr && indices != nullptr && local_embedding != nullptr &&
                            optimizer_state != nullptr);
  int cache_set_coverage                                        = 0;
  wholememory_tensor_t local_embedding_cacheline_tag_wm_tensor  = nullptr;
  wholememory_tensor_t local_embedding_cacheline_data_wm_tensor = nullptr;
  if (optimizer_state->device_cache_for_host_ != nullptr) {
    cache_set_coverage          = optimizer_state->device_cache_for_host_->get_cache_set_coverage();
    auto* local_embedding_cache = optimizer_state->device_cache_for_host_->get_cache_local_data();
    local_embedding_cacheline_tag_wm_tensor  = local_embedding_cache->cache_line_tag_;
    local_embedding_cacheline_data_wm_tensor = local_embedding_cache->cache_line_data_;
  }

  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_ops::sgd_optimizer_step(indices,
                                        grads,
                                        local_embedding,
                                        local_embedding_cacheline_tag_wm_tensor,
                                        local_embedding_cacheline_data_wm_tensor,
                                        optimizer_state->local_start_index,
                                        cache_set_coverage,
                                        weight_decay,
                                        lr,
                                        stream));
  return WHOLEMEMORY_SUCCESS;
}

class LazyAdamEmbeddingOptimizer : public embedding_optimizer_impl_base {
 public:
  LazyAdamEmbeddingOptimizer();
  void create_optimizer_states(optimizer_state_t* optimizer_state,
                               int embedding_dim) noexcept override;
  wholememory_error_code_t init_optimizer_states(
    optimizer_state_t* optimizer_state) noexcept override;
  wholememory_error_code_t step(wholememory_tensor_t indices,
                                wholememory_tensor_t grads,
                                wholememory_tensor_t local_embedding,
                                optimizer_state_t* optimizer_state,
                                float lr,
                                cudaStream_t stream) noexcept override;

 protected:
  float weight_decay = 0.0F;
  float epsilon      = 1E-8;
  float beta1        = 0.9F;
  float beta2        = 0.999F;
  float adam_w       = 0.0F;
};

LazyAdamEmbeddingOptimizer::LazyAdamEmbeddingOptimizer()
{
  name_ = "LazyAdam";
  setter_fns_.emplace(std::pair<std::string, optimizer_parameter_setter_fn_t>(
    "weight_decay", get_float_setter(&weight_decay)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("epsilon", get_float_setter(&epsilon)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("beta1", get_float_setter(&beta1)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("beta2", get_float_setter(&beta2)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("adam_w", get_float_setter(&adam_w)));
  state_names_ = {"m", "v", "beta12t", nullptr};
}

void LazyAdamEmbeddingOptimizer::create_optimizer_states(optimizer_state_t* optimizer_state,
                                                         int embedding_dim) noexcept
{
  optimizer_state->cachable_states.resize(2);
  auto& m_state = optimizer_state->cachable_states[0];
  auto& v_state = optimizer_state->cachable_states[1];

  m_state.name = "m";
  m_state.dim  = embedding_dim;

  v_state.name = "v";
  v_state.dim  = embedding_dim;

  optimizer_state->uncachable_states.resize(1);
  auto& beta12t_state = optimizer_state->uncachable_states[0];
  beta12t_state.name  = "beta12t";
  beta12t_state.dim   = 2;
  beta12t_state.dtype = WHOLEMEMORY_DT_FLOAT;
}

wholememory_error_code_t LazyAdamEmbeddingOptimizer::init_optimizer_states(
  optimizer_state_t* optimizer_state) noexcept
{
  WHOLEMEMORY_CHECK_NOTHROW(optimizer_state->cachable_states.size() == 2);
  auto& mv_state = optimizer_state->local_cachable_wm_tensor;
  zero_local_state_tensor(mv_state);
  auto& per_embedding_local_state = optimizer_state->uncachable_states[0].local_tensor;
  set_float_local_state_tensor(per_embedding_local_state, 1.0F);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t LazyAdamEmbeddingOptimizer::step(wholememory_tensor_t indices,
                                                          wholememory_tensor_t grads,
                                                          wholememory_tensor_t local_embedding,
                                                          optimizer_state_t* optimizer_state,
                                                          float lr,
                                                          cudaStream_t stream) noexcept
{
  WHOLEMEMORY_CHECK_NOTHROW(grads != nullptr && indices != nullptr && local_embedding != nullptr &&
                            optimizer_state != nullptr);
  int cache_set_coverage                                        = 0;
  wholememory_tensor_t local_embedding_cacheline_tag_wm_tensor  = nullptr;
  wholememory_tensor_t local_embedding_cacheline_data_wm_tensor = nullptr;
  wholememory_tensor_t local_state_cacheline_tag_wm_tensor      = nullptr;
  wholememory_tensor_t local_state_cacheline_data_wm_tensor     = nullptr;
  if (optimizer_state->device_cache_for_host_ != nullptr) {
    cache_set_coverage          = optimizer_state->device_cache_for_host_->get_cache_set_coverage();
    auto* local_embedding_cache = optimizer_state->device_cache_for_host_->get_cache_local_data();
    local_embedding_cacheline_tag_wm_tensor             = local_embedding_cache->cache_line_tag_;
    local_embedding_cacheline_data_wm_tensor            = local_embedding_cache->cache_line_data_;
    device_cache_for_host* state_embedding_device_cache = nullptr;
    try {
      auto* cachable_embedding_base =
        static_cast<wholememory::embedding_base*>(optimizer_state->cachable_state_embedding);
      state_embedding_device_cache =
        dynamic_cast<device_cache_for_host*>(cachable_embedding_base->get_cache_ptr());
    } catch (...) {
      WHOLEMEMORY_FAIL_NOTHROW(
        "cast from cachable_embedding_base->get_cache_ptr() to device_cache_for_host* failed.");
    }
    WHOLEMEMORY_CHECK_NOTHROW(state_embedding_device_cache != nullptr);
    auto* local_state_cache              = state_embedding_device_cache->get_cache_local_data();
    local_state_cacheline_tag_wm_tensor  = local_state_cache->cache_line_tag_;
    local_state_cacheline_data_wm_tensor = local_state_cache->cache_line_data_;
  }

  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_ops::lazy_adam_optimizer_step(indices,
                                              grads,
                                              local_embedding,
                                              local_embedding_cacheline_tag_wm_tensor,
                                              local_embedding_cacheline_data_wm_tensor,
                                              optimizer_state->local_cachable_wm_tensor,
                                              local_state_cacheline_tag_wm_tensor,
                                              local_state_cacheline_data_wm_tensor,
                                              optimizer_state->uncachable_states[0].local_tensor,
                                              optimizer_state->local_start_index,
                                              cache_set_coverage,
                                              weight_decay,
                                              epsilon,
                                              beta1,
                                              beta2,
                                              adam_w > 0.5F,
                                              lr,
                                              stream));

  return WHOLEMEMORY_SUCCESS;
}

class AdaGradEmbeddingOptimizer : public embedding_optimizer_impl_base {
 public:
  AdaGradEmbeddingOptimizer();
  void create_optimizer_states(optimizer_state_t* optimizer_state,
                               int embedding_dim) noexcept override;
  wholememory_error_code_t init_optimizer_states(
    optimizer_state_t* optimizer_state) noexcept override;
  wholememory_error_code_t step(wholememory_tensor_t indices,
                                wholememory_tensor_t grads,
                                wholememory_tensor_t local_embedding,
                                optimizer_state_t* optimizer_state,
                                float lr,
                                cudaStream_t stream) noexcept override;

 protected:
  float weight_decay = 0.0f;
  float epsilon      = 1e-8;
};
AdaGradEmbeddingOptimizer::AdaGradEmbeddingOptimizer()
{
  name_ = "AdaGrad";
  setter_fns_.emplace(std::pair<std::string, optimizer_parameter_setter_fn_t>(
    "weight_decay", get_float_setter(&weight_decay)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("epsilon", get_float_setter(&epsilon)));
  state_names_ = {"state_sum", nullptr};
}

void AdaGradEmbeddingOptimizer::create_optimizer_states(optimizer_state_t* optimizer_state,
                                                        int embedding_dim) noexcept
{
  optimizer_state->cachable_states.resize(1);
  auto& state_sum_state = optimizer_state->cachable_states[0];
  state_sum_state.name  = "state_sum";
  state_sum_state.dim   = embedding_dim;
}

wholememory_error_code_t AdaGradEmbeddingOptimizer::init_optimizer_states(
  optimizer_state_t* optimizer_state) noexcept
{
  WHOLEMEMORY_CHECK_NOTHROW(optimizer_state->cachable_states.size() == 1);
  auto& state_sum_state = optimizer_state->local_cachable_wm_tensor;
  zero_local_state_tensor(state_sum_state);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t AdaGradEmbeddingOptimizer::step(wholememory_tensor_t indices,
                                                         wholememory_tensor_t grads,
                                                         wholememory_tensor_t local_embedding,
                                                         optimizer_state_t* optimizer_state,
                                                         float lr,
                                                         cudaStream_t stream) noexcept
{
  WHOLEMEMORY_CHECK_NOTHROW(grads != nullptr && indices != nullptr && local_embedding != nullptr &&
                            optimizer_state != nullptr);
  int cache_set_coverage                                        = 0;
  wholememory_tensor_t local_embedding_cacheline_tag_wm_tensor  = nullptr;
  wholememory_tensor_t local_embedding_cacheline_data_wm_tensor = nullptr;
  wholememory_tensor_t local_state_cacheline_tag_wm_tensor      = nullptr;
  wholememory_tensor_t local_state_cacheline_data_wm_tensor     = nullptr;
  if (optimizer_state->device_cache_for_host_ != nullptr) {
    cache_set_coverage          = optimizer_state->device_cache_for_host_->get_cache_set_coverage();
    auto* local_embedding_cache = optimizer_state->device_cache_for_host_->get_cache_local_data();
    local_embedding_cacheline_tag_wm_tensor             = local_embedding_cache->cache_line_tag_;
    local_embedding_cacheline_data_wm_tensor            = local_embedding_cache->cache_line_data_;
    device_cache_for_host* state_embedding_device_cache = nullptr;
    try {
      auto* cachable_embedding_base =
        static_cast<wholememory::embedding_base*>(optimizer_state->cachable_state_embedding);
      state_embedding_device_cache =
        dynamic_cast<device_cache_for_host*>(cachable_embedding_base->get_cache_ptr());
    } catch (...) {
      WHOLEMEMORY_FAIL_NOTHROW(
        "cast from cachable_embedding_base->get_cache_ptr() to device_cache_for_host* failed.");
    }
    WHOLEMEMORY_CHECK_NOTHROW(state_embedding_device_cache != nullptr);
    auto* local_state_cache              = state_embedding_device_cache->get_cache_local_data();
    local_state_cacheline_tag_wm_tensor  = local_state_cache->cache_line_tag_;
    local_state_cacheline_data_wm_tensor = local_state_cache->cache_line_data_;
  }
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_ops::ada_grad_optimizer_step(indices,
                                             grads,
                                             local_embedding,
                                             local_embedding_cacheline_tag_wm_tensor,
                                             local_embedding_cacheline_data_wm_tensor,
                                             optimizer_state->local_cachable_wm_tensor,
                                             local_state_cacheline_tag_wm_tensor,
                                             local_state_cacheline_data_wm_tensor,
                                             optimizer_state->local_start_index,
                                             cache_set_coverage,
                                             weight_decay,
                                             epsilon,
                                             lr,
                                             stream));

  return WHOLEMEMORY_SUCCESS;
}

class RMSPropEmbeddingOptimizer : public embedding_optimizer_impl_base {
 public:
  RMSPropEmbeddingOptimizer();
  void create_optimizer_states(optimizer_state_t* optimizer_state,
                               int embedding_dim) noexcept override;
  wholememory_error_code_t init_optimizer_states(
    optimizer_state_t* optimizer_state) noexcept override;
  wholememory_error_code_t step(wholememory_tensor_t indices,
                                wholememory_tensor_t grads,
                                wholememory_tensor_t local_embedding,
                                optimizer_state_t* optimizer_state,
                                float lr,
                                cudaStream_t stream) noexcept override;

 protected:
  float weight_decay = 0.0f;
  float epsilon      = 1e-8;
  float alpha        = 0.99;
};

RMSPropEmbeddingOptimizer::RMSPropEmbeddingOptimizer()
{
  name_ = "RMSProp";
  setter_fns_.emplace(std::pair<std::string, optimizer_parameter_setter_fn_t>(
    "weight_decay", get_float_setter(&weight_decay)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("epsilon", get_float_setter(&epsilon)));
  setter_fns_.emplace(
    std::pair<std::string, optimizer_parameter_setter_fn_t>("alpha", get_float_setter(&alpha)));
  state_names_ = {"v", nullptr};
}

void RMSPropEmbeddingOptimizer::create_optimizer_states(optimizer_state_t* optimizer_state,
                                                        int embedding_dim) noexcept
{
  optimizer_state->cachable_states.resize(1);
  auto& v_state = optimizer_state->cachable_states[0];
  v_state.name  = "v";
  v_state.dim   = embedding_dim;
}

wholememory_error_code_t RMSPropEmbeddingOptimizer::init_optimizer_states(
  optimizer_state_t* optimizer_state) noexcept
{
  WHOLEMEMORY_CHECK_NOTHROW(optimizer_state->cachable_states.size() == 1);
  auto& v_state = optimizer_state->local_cachable_wm_tensor;
  zero_local_state_tensor(v_state);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t RMSPropEmbeddingOptimizer::step(wholememory_tensor_t indices,
                                                         wholememory_tensor_t grads,
                                                         wholememory_tensor_t local_embedding,
                                                         optimizer_state_t* optimizer_state,
                                                         float lr,
                                                         cudaStream_t stream) noexcept
{
  WHOLEMEMORY_CHECK_NOTHROW(grads != nullptr && indices != nullptr && local_embedding != nullptr &&
                            optimizer_state != nullptr);
  int cache_set_coverage                                        = 0;
  wholememory_tensor_t local_embedding_cacheline_tag_wm_tensor  = nullptr;
  wholememory_tensor_t local_embedding_cacheline_data_wm_tensor = nullptr;
  wholememory_tensor_t local_state_cacheline_tag_wm_tensor      = nullptr;
  wholememory_tensor_t local_state_cacheline_data_wm_tensor     = nullptr;
  if (optimizer_state->device_cache_for_host_ != nullptr) {
    cache_set_coverage          = optimizer_state->device_cache_for_host_->get_cache_set_coverage();
    auto* local_embedding_cache = optimizer_state->device_cache_for_host_->get_cache_local_data();
    local_embedding_cacheline_tag_wm_tensor             = local_embedding_cache->cache_line_tag_;
    local_embedding_cacheline_data_wm_tensor            = local_embedding_cache->cache_line_data_;
    device_cache_for_host* state_embedding_device_cache = nullptr;
    try {
      auto* cachable_embedding_base =
        static_cast<wholememory::embedding_base*>(optimizer_state->cachable_state_embedding);
      state_embedding_device_cache =
        dynamic_cast<device_cache_for_host*>(cachable_embedding_base->get_cache_ptr());
    } catch (...) {
      WHOLEMEMORY_FAIL_NOTHROW(
        "cast from cachable_embedding_base->get_cache_ptr() to device_cache_for_host* failed.");
    }
    WHOLEMEMORY_CHECK_NOTHROW(state_embedding_device_cache != nullptr);
    auto* local_state_cache              = state_embedding_device_cache->get_cache_local_data();
    local_state_cacheline_tag_wm_tensor  = local_state_cache->cache_line_tag_;
    local_state_cacheline_data_wm_tensor = local_state_cache->cache_line_data_;
  }
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_ops::rms_prop_optimizer_step(indices,
                                             grads,
                                             local_embedding,
                                             local_embedding_cacheline_tag_wm_tensor,
                                             local_embedding_cacheline_data_wm_tensor,
                                             optimizer_state->local_cachable_wm_tensor,
                                             local_state_cacheline_tag_wm_tensor,
                                             local_state_cacheline_data_wm_tensor,
                                             optimizer_state->local_start_index,
                                             cache_set_coverage,
                                             weight_decay,
                                             epsilon,
                                             alpha,
                                             lr,
                                             stream));

  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t create_embedding_optimizer(
  wholememory_embedding_optimizer_t* optimizer,
  wholememory_optimizer_type_t optimizer_type) noexcept
{
  embedding_optimizer_impl_base* optimizer_impl = nullptr;
  try {
    switch (optimizer_type) {
      case WHOLEMEMORY_OPT_SGD: {
        optimizer_impl = new SGDEmbeddingOptimizer();
        break;
      }
      case WHOLEMEMORY_OPT_LAZY_ADAM: {
        optimizer_impl = new LazyAdamEmbeddingOptimizer();
        break;
      }
      case WHOLEMEMORY_OPT_ADAGRAD: {
        optimizer_impl = new AdaGradEmbeddingOptimizer();
        break;
      }
      case WHOLEMEMORY_OPT_RMSPROP: {
        optimizer_impl = new RMSPropEmbeddingOptimizer();
        break;
      }
      default: {
        return WHOLEMEMORY_NOT_IMPLEMENTED;
      }
    }
  } catch (...) {
    WHOLEMEMORY_ERROR("create optimizer failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  *optimizer = static_cast<wholememory_embedding_optimizer_t>(optimizer_impl);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t optimizer_set_parameter(wholememory_embedding_optimizer_t optimizer,
                                                 const char* parameter_name,
                                                 void* value) noexcept
{
  if (optimizer == nullptr) {
    WHOLEMEMORY_ERROR("Input optimizer is nullptr.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto* optimizer_impl = static_cast<embedding_optimizer_impl_base*>(optimizer);
  return optimizer_impl->set_parameter(parameter_name, value);
}

void destroy_embedding_optimizer(wholememory_embedding_optimizer_t optimizer) noexcept
{
  auto* optimizer_impl = static_cast<embedding_optimizer_impl_base*>(optimizer);
  delete optimizer_impl;
}

}  // namespace wholememory
