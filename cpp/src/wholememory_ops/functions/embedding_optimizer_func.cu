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
#include "embedding_optimizer_func.h"

#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"
#include "wholememory/integer_utils.hpp"
#include "wholememory_ops/functions/embedding_cache_func.cuh"
#include "wholememory_ops/register.hpp"

#include <wholememory/device_reference.cuh>

namespace wholememory_ops {

__global__ void set_float_kernel(float* data_ptr, float value, size_t elt_count)
{
  int64_t idx = blockIdx.x;
  idx *= blockDim.x;
  idx += threadIdx.x;
  if (idx >= elt_count) return;
  data_ptr[idx] = value;
}

void set_memory_to_float_value(float* data_ptr, float value, size_t elt_count, cudaStream_t stream)
{
  const int thread_count = 128;
  int block_count        = wholememory::div_rounding_up_safe<int64_t>(elt_count, thread_count);
  set_float_kernel<<<block_count, thread_count, 0, stream>>>(data_ptr, value, elt_count);
  WM_CUDA_CHECK_NO_THROW(cudaGetLastError());
}

template <int PerElementCount = 0>
static void check_optimizer_inputs(wholememory_tensor_t indices,
                                   wholememory_tensor_t grads,
                                   wholememory_tensor_t local_embedding,
                                   wholememory_tensor_t local_embedding_cache_tag,
                                   wholememory_tensor_t local_embedding_cache_data,
                                   wholememory_tensor_t per_element_local_state,
                                   wholememory_tensor_t per_element_local_cache_tag,
                                   wholememory_tensor_t per_element_local_cache_data,
                                   int cache_set_coverage)
{
  WHOLEMEMORY_CHECK_NOTHROW(indices != nullptr && grads != nullptr && local_embedding != nullptr);
  if (cache_set_coverage > 0) {
    WHOLEMEMORY_CHECK_NOTHROW(local_embedding_cache_tag != nullptr &&
                              local_embedding_cache_data != nullptr);
  }
  if (PerElementCount > 0) {
    WHOLEMEMORY_CHECK_NOTHROW(per_element_local_state != nullptr);
    if (cache_set_coverage > 0) {
      WHOLEMEMORY_CHECK_NOTHROW(per_element_local_cache_tag != nullptr &&
                                per_element_local_cache_data != nullptr);
    }
  }
  auto* indices_desc = wholememory_tensor_get_tensor_description(indices);
  WHOLEMEMORY_CHECK_NOTHROW(indices_desc->dim == 1);
  WHOLEMEMORY_CHECK_NOTHROW(indices_desc->dtype == WHOLEMEMORY_DT_INT ||
                            indices_desc->dtype == WHOLEMEMORY_DT_INT64);
  WHOLEMEMORY_CHECK_NOTHROW(indices_desc->storage_offset == 0);

  auto* grads_desc = wholememory_tensor_get_tensor_description(grads);
  WHOLEMEMORY_CHECK_NOTHROW(grads_desc->dim == 2);
  WHOLEMEMORY_CHECK_NOTHROW(grads_desc->dtype == WHOLEMEMORY_DT_FLOAT);
  WHOLEMEMORY_CHECK_NOTHROW(grads_desc->storage_offset == 0);
  WHOLEMEMORY_CHECK_NOTHROW(grads_desc->sizes[0] == indices_desc->sizes[0]);

  int embedding_dim = grads_desc->sizes[1];

  auto* local_embedding_desc = wholememory_tensor_get_tensor_description(local_embedding);
  WHOLEMEMORY_CHECK_NOTHROW(local_embedding_desc->dim == 2);
  WHOLEMEMORY_CHECK_NOTHROW(local_embedding_desc->dtype == WHOLEMEMORY_DT_FLOAT);
  WHOLEMEMORY_CHECK_NOTHROW(local_embedding_desc->storage_offset == 0);

  int padded_embedding_dim            = local_embedding_desc->strides[0];
  int64_t local_embedding_entry_count = local_embedding_desc->sizes[0];

  // WHOLEMEMORY_CHECK_NOTHROW(embedding_dim <= padded_embedding_dim);
  // if (wholememory::round_up_unsafe<int>(embedding_dim, 4) != padded_embedding_dim) {
  //   WHOLEMEMORY_FAIL_NOTHROW("embedding_dim=%d, but padded_embedding_dim=%d", embedding_dim,
  //   padded_embedding_dim);
  // }
  WHOLEMEMORY_CHECK_NOTHROW(wholememory::round_up_unsafe<int>(embedding_dim, 4) ==
                            padded_embedding_dim);

  if (cache_set_coverage > 0) {
    auto* local_embedding_cache_tag_desc =
      wholememory_tensor_get_tensor_description(local_embedding_cache_tag);
    WHOLEMEMORY_CHECK_NOTHROW(local_embedding_cache_tag_desc->dim == 2);
    WHOLEMEMORY_CHECK_NOTHROW(local_embedding_cache_tag_desc->dtype == WHOLEMEMORY_DT_INT16);
    WHOLEMEMORY_CHECK_NOTHROW(local_embedding_cache_tag_desc->storage_offset == 0);
    WHOLEMEMORY_CHECK_NOTHROW(local_embedding_cache_tag_desc->sizes[1] == 32);
    int64_t local_cache_set_count = local_embedding_cache_tag_desc->sizes[0];
    WHOLEMEMORY_CHECK_NOTHROW(local_cache_set_count * cache_set_coverage >=
                              local_embedding_entry_count);
    auto* local_embedding_cache_data_desc =
      wholememory_tensor_get_tensor_description(local_embedding_cache_data);
    WHOLEMEMORY_CHECK_NOTHROW(local_embedding_cache_data_desc->dim == 2);
    WHOLEMEMORY_CHECK_NOTHROW(local_embedding_cache_data_desc->dtype == WHOLEMEMORY_DT_FLOAT);
    WHOLEMEMORY_CHECK_NOTHROW(local_embedding_cache_data_desc->storage_offset == 0);
    WHOLEMEMORY_CHECK_NOTHROW(local_embedding_cache_data_desc->strides[0] == padded_embedding_dim);
    WHOLEMEMORY_CHECK_NOTHROW(local_embedding_cache_data_desc->sizes[0] ==
                              local_cache_set_count * 32);
  }

  if (PerElementCount > 0) {
    auto* local_per_element_desc =
      wholememory_tensor_get_tensor_description(per_element_local_state);
    WHOLEMEMORY_CHECK_NOTHROW(local_per_element_desc->dim == 2);
    WHOLEMEMORY_CHECK_NOTHROW(local_per_element_desc->dtype == WHOLEMEMORY_DT_FLOAT);
    WHOLEMEMORY_CHECK_NOTHROW(local_per_element_desc->storage_offset == 0);
    WHOLEMEMORY_CHECK_NOTHROW(local_per_element_desc->sizes[0] == local_embedding_entry_count);
    WHOLEMEMORY_CHECK_NOTHROW(local_per_element_desc->sizes[1] ==
                              PerElementCount * (int64_t)padded_embedding_dim);
    int64_t local_per_element_dim = local_per_element_desc->sizes[1];
    if (cache_set_coverage > 0) {
      auto* per_element_local_cache_tag_desc =
        wholememory_tensor_get_tensor_description(per_element_local_cache_tag);
      WHOLEMEMORY_CHECK_NOTHROW(per_element_local_cache_tag_desc->dim == 2);
      WHOLEMEMORY_CHECK_NOTHROW(per_element_local_cache_tag_desc->dtype == WHOLEMEMORY_DT_INT16);
      WHOLEMEMORY_CHECK_NOTHROW(per_element_local_cache_tag_desc->storage_offset == 0);
      WHOLEMEMORY_CHECK_NOTHROW(per_element_local_cache_tag_desc->sizes[1] == 32);
      int64_t local_cache_set_count = per_element_local_cache_tag_desc->sizes[0];
      WHOLEMEMORY_CHECK_NOTHROW(local_cache_set_count * cache_set_coverage >=
                                local_embedding_entry_count);
      auto* per_element_local_cache_data_desc =
        wholememory_tensor_get_tensor_description(per_element_local_cache_data);
      WHOLEMEMORY_CHECK_NOTHROW(per_element_local_cache_data_desc->dim == 2);
      WHOLEMEMORY_CHECK_NOTHROW(per_element_local_cache_data_desc->dtype == WHOLEMEMORY_DT_FLOAT);
      WHOLEMEMORY_CHECK_NOTHROW(per_element_local_cache_data_desc->storage_offset == 0);
      WHOLEMEMORY_CHECK_NOTHROW(per_element_local_cache_data_desc->sizes[1] ==
                                local_per_element_dim);
      WHOLEMEMORY_CHECK_NOTHROW(per_element_local_cache_data_desc->sizes[0] ==
                                local_cache_set_count * 32);
    }
  }
}

template <bool UseCache>
static __device__ __forceinline__ float* optimizer_get_ptr_from_cache(float* local_ptr,
                                                                      uint16_t* local_cache_tag_ptr,
                                                                      float* local_cache_data_ptr,
                                                                      int64_t indice_in_local_rank,
                                                                      int embedding_stride,
                                                                      int cache_set_coverage)
{
  float* non_cached_ptr = local_ptr + indice_in_local_rank * embedding_stride;
  if (!UseCache) { return non_cached_ptr; }
  int local_cache_set_id = indice_in_local_rank / cache_set_coverage;
  int local_id =
    indice_in_local_rank - static_cast<int64_t>(local_cache_set_id) * cache_set_coverage;
  local_cache_tag_ptr += static_cast<int64_t>(local_cache_set_id) * CacheLineInfo::kCacheSetSize;
  CacheLineInfo cache_line_info;
  cache_line_info.LoadTag(local_cache_tag_ptr);
  int const cached_line_id = cache_line_info.KeyIndexSync(local_id);
  if (cached_line_id == -1) { return non_cached_ptr; }
  cache_line_info.SetModified(local_id);
  if (threadIdx.x == cached_line_id) { local_cache_tag_ptr[threadIdx.x] = cache_line_info.tag_; }
  return local_cache_data_ptr +
         (static_cast<int64_t>(local_cache_set_id) * CacheLineInfo::kCacheSetSize +
          cached_line_id) *
           embedding_stride;
}

template <typename IndiceT, bool UseCache>
__global__ void sgd_optimizer_step_kernel(const IndiceT* indices_ptr,
                                          const float* grads_ptr,
                                          float* local_embedding_ptr,
                                          uint16_t* local_embedding_cache_tag_ptr,
                                          float* local_embedding_cache_data_ptr,
                                          int64_t local_entry_offset,
                                          int embedding_dim,
                                          int grad_stride,
                                          int local_embedding_stride,
                                          int cache_set_coverage,
                                          float weight_decay,
                                          float lr)
{
  int64_t block_idx = blockIdx.x;
  auto indice       = indices_ptr[block_idx];
  grads_ptr += block_idx * grad_stride;
  IndiceT local_rank_indice = indice - local_entry_offset;

  __shared__ float* s_embedding_ptr;

  float* embedding_ptr;
  if (threadIdx.x < 32) {
    embedding_ptr = optimizer_get_ptr_from_cache<UseCache>(local_embedding_ptr,
                                                           local_embedding_cache_tag_ptr,
                                                           local_embedding_cache_data_ptr,
                                                           local_rank_indice,
                                                           local_embedding_stride,
                                                           cache_set_coverage);
    if (threadIdx.x == 0) { s_embedding_ptr = embedding_ptr; }
  }
  __syncthreads();
  embedding_ptr = s_embedding_ptr;

  int loop_start_idx = 0;
  for (; loop_start_idx < embedding_dim; loop_start_idx += blockDim.x) {
    int local_dim_idx = threadIdx.x;
    float grad_value  = 0.0f;
    int embedding_idx = local_dim_idx + loop_start_idx;
    if (embedding_idx >= embedding_dim) { break; }
    grad_value            = grads_ptr[embedding_idx];
    float embedding_value = embedding_ptr[embedding_idx];
    grad_value += weight_decay * embedding_value;
    embedding_value -= lr * grad_value;
    embedding_ptr[embedding_idx] = embedding_value;
  }
}

template <typename IndiceT>
void sgd_optimizer_step_temp_func(const void* indices_ptr,
                                  const float* grads_ptr,
                                  float* local_embedding_ptr,
                                  uint16_t* local_embedding_cache_tag_ptr,
                                  float* local_embedding_cache_data_ptr,
                                  int64_t local_entry_offset,
                                  int indice_count,
                                  int embedding_dim,
                                  int grad_stride,
                                  int local_embedding_stride,
                                  int cache_set_coverage,
                                  float weight_decay,
                                  float lr,
                                  cudaStream_t stream)
{
  const IndiceT* typed_indices_ptr = static_cast<const IndiceT*>(indices_ptr);
  int block_count                  = indice_count;
  if (block_count == 0) return;
  int thread_count = wholememory::div_rounding_up_unsafe(embedding_dim, 4);
  if (thread_count > 512) thread_count = 512;
  if (thread_count < 32) thread_count = 32;
  auto func_ptr = sgd_optimizer_step_kernel<IndiceT, false>;
  if (cache_set_coverage > 0) { func_ptr = sgd_optimizer_step_kernel<IndiceT, true>; }
  func_ptr<<<block_count, thread_count, 0, stream>>>(typed_indices_ptr,
                                                     grads_ptr,
                                                     local_embedding_ptr,
                                                     local_embedding_cache_tag_ptr,
                                                     local_embedding_cache_data_ptr,
                                                     local_entry_offset,
                                                     embedding_dim,
                                                     grad_stride,
                                                     local_embedding_stride,
                                                     cache_set_coverage,
                                                     weight_decay,
                                                     lr);
  WM_CUDA_CHECK(cudaGetLastError());
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
}

REGISTER_DISPATCH_ONE_TYPE(SGDOptimizerStepTempFunc, sgd_optimizer_step_temp_func, SINT3264)

wholememory_error_code_t sgd_optimizer_step(wholememory_tensor_t indices,
                                            wholememory_tensor_t grads,
                                            wholememory_tensor_t local_embedding,
                                            wholememory_tensor_t local_embedding_cache_tag,
                                            wholememory_tensor_t local_embedding_cache_data,
                                            int64_t local_entry_offset,
                                            int cache_set_coverage,
                                            float weight_decay,
                                            float lr,
                                            cudaStream_t stream)
{
  try {
    check_optimizer_inputs<0>(indices,
                              grads,
                              local_embedding,
                              local_embedding_cache_tag,
                              local_embedding_cache_data,
                              nullptr,
                              nullptr,
                              nullptr,
                              cache_set_coverage);
    auto* indice_desc          = wholememory_tensor_get_tensor_description(indices);
    auto* grads_desc           = wholememory_tensor_get_tensor_description(grads);
    auto* local_embedding_desc = wholememory_tensor_get_tensor_description(local_embedding);

    uint16_t* local_embedding_cache_tag_pr = nullptr;
    float* local_embedding_cache_data_ptr  = nullptr;
    if (cache_set_coverage > 0) {
      local_embedding_cache_tag_pr =
        static_cast<uint16_t*>(wholememory_tensor_get_data_pointer(local_embedding_cache_tag));
      local_embedding_cache_data_ptr =
        static_cast<float*>(wholememory_tensor_get_data_pointer(local_embedding_cache_data));
    }

    DISPATCH_ONE_TYPE(indice_desc->dtype,
                      SGDOptimizerStepTempFunc,
                      wholememory_tensor_get_data_pointer(indices),
                      static_cast<float*>(wholememory_tensor_get_data_pointer(grads)),
                      static_cast<float*>(wholememory_tensor_get_data_pointer(local_embedding)),
                      local_embedding_cache_tag_pr,
                      local_embedding_cache_data_ptr,
                      local_entry_offset,
                      indice_desc->sizes[0],
                      grads_desc->sizes[1],
                      grads_desc->strides[0],
                      local_embedding_desc->strides[0],
                      cache_set_coverage,
                      weight_decay,
                      lr,
                      stream);
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("%s", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("%s", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (...) {
    WHOLEMEMORY_ERROR("File %s, line %d, Unknown error", __FILE__, __LINE__);
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

template <typename IndiceT, bool UseCache, bool AdamW = false>
__global__ void lazy_adam_optimizer_step_kernel(const IndiceT* indices_ptr,
                                                const float* grads_ptr,
                                                float* local_embedding_ptr,
                                                uint16_t* local_embedding_cache_tag_ptr,
                                                float* local_embedding_cache_data_ptr,
                                                float* per_element_local_embedding_ptr,
                                                uint16_t* per_element_local_cache_tag_ptr,
                                                float* per_element_local_cache_data_ptr,
                                                float* per_embedding_state_local_ptr,
                                                int64_t local_entry_offset,
                                                int embedding_dim,
                                                int grad_stride,
                                                int local_embedding_stride,
                                                int cache_set_coverage,
                                                float weight_decay,
                                                float epsilon,
                                                float beta1,
                                                float beta2,
                                                float lr)
{
  int64_t block_idx = blockIdx.x;
  auto indice       = indices_ptr[block_idx];
  grads_ptr += block_idx * grad_stride;
  IndiceT local_rank_indice = indice - local_entry_offset;
  per_embedding_state_local_ptr += static_cast<int64_t>(local_rank_indice) * 2;

  __shared__ float *s_embedding_ptr, *s_per_element_ptr;

  float *embedding_ptr, *per_element_ptr;
  if (threadIdx.x < 32) {
    embedding_ptr   = optimizer_get_ptr_from_cache<UseCache>(local_embedding_ptr,
                                                           local_embedding_cache_tag_ptr,
                                                           local_embedding_cache_data_ptr,
                                                           local_rank_indice,
                                                           local_embedding_stride,
                                                           cache_set_coverage);
    per_element_ptr = optimizer_get_ptr_from_cache<UseCache>(per_element_local_embedding_ptr,
                                                             per_element_local_cache_tag_ptr,
                                                             per_element_local_cache_data_ptr,
                                                             local_rank_indice,
                                                             local_embedding_stride * 2,
                                                             cache_set_coverage);
    if (threadIdx.x == 0) {
      s_embedding_ptr   = embedding_ptr;
      s_per_element_ptr = per_element_ptr;
    }
  }
  __syncthreads();
  embedding_ptr   = s_embedding_ptr;
  per_element_ptr = s_per_element_ptr;

  float* m_ptr = per_element_ptr;
  float* v_ptr = per_element_ptr + local_embedding_stride;

  float beta1t = per_embedding_state_local_ptr[0];
  float beta2t = per_embedding_state_local_ptr[1];
  beta1t *= beta1;
  beta2t *= beta2;

  int loop_start_idx = 0;
  for (; loop_start_idx < embedding_dim; loop_start_idx += blockDim.x) {
    int local_dim_idx = threadIdx.x;
    float grad_value  = 0.0f;
    int embedding_idx = local_dim_idx + loop_start_idx;
    if (embedding_idx >= embedding_dim) { break; }
    grad_value            = grads_ptr[local_dim_idx + loop_start_idx];
    float embedding_value = embedding_ptr[embedding_idx];
    if (AdamW) {
      embedding_value -= lr * weight_decay * embedding_value;
    } else {
      grad_value = grad_value + weight_decay * embedding_value;
    }
    float m                      = m_ptr[embedding_idx];
    float v                      = v_ptr[embedding_idx];
    m                            = beta1 * m + (1 - beta1) * grad_value;
    v                            = beta2 * v + (1 - beta2) * grad_value * grad_value;
    float mhat                   = m / (1 - beta1t);
    float vhat                   = v / (1 - beta2t);
    embedding_value              = embedding_value - lr * mhat / (sqrtf(vhat) + epsilon);
    m_ptr[embedding_idx]         = m;
    v_ptr[embedding_idx]         = v;
    embedding_ptr[embedding_idx] = embedding_value;
  }
  if (threadIdx.x == 0) {
    per_embedding_state_local_ptr[0] = beta1t;
    per_embedding_state_local_ptr[1] = beta2t;
  }
}

template <typename IndiceT>
void lazy_adam_optimizer_step_temp_func(const void* indices_ptr,
                                        const float* grads_ptr,
                                        float* local_embedding_ptr,
                                        uint16_t* local_embedding_cache_tag_ptr,
                                        float* local_embedding_cache_data_ptr,
                                        float* per_element_local_embedding_ptr,
                                        uint16_t* per_element_local_cache_tag_ptr,
                                        float* per_element_local_cache_data_ptr,
                                        float* per_embedding_state_local_ptr,
                                        int64_t local_entry_offset,
                                        int indice_count,
                                        int embedding_dim,
                                        int grad_stride,
                                        int local_embedding_stride,
                                        int cache_set_coverage,
                                        float weight_decay,
                                        float epsilon,
                                        float beta1,
                                        float beta2,
                                        bool adam_w,
                                        float lr,
                                        cudaStream_t stream)
{
  const IndiceT* typed_indices_ptr = static_cast<const IndiceT*>(indices_ptr);
  int block_count                  = indice_count;
  if (block_count == 0) return;
  int thread_count = wholememory::div_rounding_up_unsafe(embedding_dim, 4);
  if (thread_count > 512) thread_count = 512;
  if (thread_count < 32) thread_count = 32;
  auto func_ptr = lazy_adam_optimizer_step_kernel<IndiceT, false>;
  if (cache_set_coverage > 0) {
    if (adam_w == false) {
      func_ptr = lazy_adam_optimizer_step_kernel<IndiceT, true, false>;
    } else {
      func_ptr = lazy_adam_optimizer_step_kernel<IndiceT, true, true>;
    }
  } else {
    if (adam_w == false) {
      func_ptr = lazy_adam_optimizer_step_kernel<IndiceT, false, false>;
    } else {
      func_ptr = lazy_adam_optimizer_step_kernel<IndiceT, false, true>;
    }
  }
  func_ptr<<<block_count, thread_count, 0, stream>>>(typed_indices_ptr,
                                                     grads_ptr,
                                                     local_embedding_ptr,
                                                     local_embedding_cache_tag_ptr,
                                                     local_embedding_cache_data_ptr,
                                                     per_element_local_embedding_ptr,
                                                     per_element_local_cache_tag_ptr,
                                                     per_element_local_cache_data_ptr,
                                                     per_embedding_state_local_ptr,
                                                     local_entry_offset,
                                                     embedding_dim,
                                                     grad_stride,
                                                     local_embedding_stride,
                                                     cache_set_coverage,
                                                     weight_decay,
                                                     epsilon,
                                                     beta1,
                                                     beta2,
                                                     lr);
  WM_CUDA_CHECK(cudaGetLastError());
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
}

REGISTER_DISPATCH_ONE_TYPE(LazyAdamOptimizerStepTempFunc,
                           lazy_adam_optimizer_step_temp_func,
                           SINT3264)

wholememory_error_code_t lazy_adam_optimizer_step(wholememory_tensor_t indices,
                                                  wholememory_tensor_t grads,
                                                  wholememory_tensor_t local_embedding,
                                                  wholememory_tensor_t local_embedding_cache_tag,
                                                  wholememory_tensor_t local_embedding_cache_data,
                                                  wholememory_tensor_t per_element_local_state,
                                                  wholememory_tensor_t per_element_local_cache_tag,
                                                  wholememory_tensor_t per_element_local_cache_data,
                                                  wholememory_tensor_t per_embedding_local_state,
                                                  int64_t local_entry_offset,
                                                  int cache_set_coverage,
                                                  float weight_decay,
                                                  float epsilon,
                                                  float beta1,
                                                  float beta2,
                                                  bool adam_w,
                                                  float lr,
                                                  cudaStream_t stream)
{
  try {
    check_optimizer_inputs<2>(indices,
                              grads,
                              local_embedding,
                              local_embedding_cache_tag,
                              local_embedding_cache_data,
                              per_element_local_state,
                              per_element_local_cache_tag,
                              per_element_local_cache_data,
                              cache_set_coverage);
    auto* indice_desc          = wholememory_tensor_get_tensor_description(indices);
    auto* grads_desc           = wholememory_tensor_get_tensor_description(grads);
    auto* local_embedding_desc = wholememory_tensor_get_tensor_description(local_embedding);
    int64_t local_embedding_entry_count = local_embedding_desc->sizes[0];
    WHOLEMEMORY_CHECK_NOTHROW(per_embedding_local_state != nullptr);
    auto* per_embedding_local_state_desc =
      wholememory_tensor_get_tensor_description(per_embedding_local_state);
    WHOLEMEMORY_CHECK_NOTHROW(per_embedding_local_state_desc->dim == 2);
    WHOLEMEMORY_CHECK_NOTHROW(per_embedding_local_state_desc->dtype == WHOLEMEMORY_DT_FLOAT);
    WHOLEMEMORY_CHECK_NOTHROW(per_embedding_local_state_desc->storage_offset == 0);
    WHOLEMEMORY_CHECK_NOTHROW(per_embedding_local_state_desc->sizes[1] == 2);
    if (local_embedding_entry_count != per_embedding_local_state_desc->sizes[0]) {
      WHOLEMEMORY_FAIL_NOTHROW(
        "local_embedding_entry_count=%ld, but per_embedding_local_state_desc->sizes[0]=%ld",
        local_embedding_entry_count,
        per_embedding_local_state_desc->sizes[0]);
    }
    WHOLEMEMORY_CHECK_NOTHROW(local_embedding_entry_count ==
                              per_embedding_local_state_desc->sizes[0]);

    uint16_t* local_embedding_cache_tag_pr    = nullptr;
    float* local_embedding_cache_data_ptr     = nullptr;
    uint16_t* per_element_local_cache_tag_ptr = nullptr;
    float* per_element_local_cache_data_ptr   = nullptr;
    if (cache_set_coverage > 0) {
      local_embedding_cache_tag_pr =
        static_cast<uint16_t*>(wholememory_tensor_get_data_pointer(local_embedding_cache_tag));
      local_embedding_cache_data_ptr =
        static_cast<float*>(wholememory_tensor_get_data_pointer(local_embedding_cache_data));
      per_element_local_cache_tag_ptr =
        static_cast<uint16_t*>(wholememory_tensor_get_data_pointer(per_element_local_cache_tag));
      per_element_local_cache_data_ptr =
        static_cast<float*>(wholememory_tensor_get_data_pointer(per_element_local_cache_data));
    }

    DISPATCH_ONE_TYPE(
      indice_desc->dtype,
      LazyAdamOptimizerStepTempFunc,
      wholememory_tensor_get_data_pointer(indices),
      static_cast<float*>(wholememory_tensor_get_data_pointer(grads)),
      static_cast<float*>(wholememory_tensor_get_data_pointer(local_embedding)),
      local_embedding_cache_tag_pr,
      local_embedding_cache_data_ptr,
      static_cast<float*>(wholememory_tensor_get_data_pointer(per_element_local_state)),
      per_element_local_cache_tag_ptr,
      per_element_local_cache_data_ptr,
      static_cast<float*>(wholememory_tensor_get_data_pointer(per_embedding_local_state)),
      local_entry_offset,
      indice_desc->sizes[0],
      grads_desc->sizes[1],
      grads_desc->strides[0],
      local_embedding_desc->strides[0],
      cache_set_coverage,
      weight_decay,
      epsilon,
      beta1,
      beta2,
      adam_w,
      lr,
      stream);
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("%s", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("%s", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (...) {
    WHOLEMEMORY_ERROR("File %s, line %d, Unknown error", __FILE__, __LINE__);
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

template <typename IndiceT, bool UseCache>
__global__ void ada_grad_optimizer_step_kernel(const IndiceT* indices_ptr,
                                               const float* grads_ptr,
                                               float* local_embedding_ptr,
                                               uint16_t* local_embedding_cache_tag_ptr,
                                               float* local_embedding_cache_data_ptr,
                                               float* per_element_local_embedding_ptr,
                                               uint16_t* per_element_local_cache_tag_ptr,
                                               float* per_element_local_cache_data_ptr,
                                               int64_t local_entry_offset,
                                               int embedding_dim,
                                               int grad_stride,
                                               int local_embedding_stride,
                                               int cache_set_coverage,
                                               float weight_decay,
                                               float epsilon,
                                               float lr)
{
  int64_t block_idx = blockIdx.x;
  auto indice       = indices_ptr[block_idx];
  grads_ptr += block_idx * grad_stride;
  IndiceT local_rank_indice = indice - local_entry_offset;

  __shared__ float *s_embedding_ptr, *s_per_element_ptr;

  float *embedding_ptr, *per_element_ptr;
  if (threadIdx.x < 32) {
    embedding_ptr   = optimizer_get_ptr_from_cache<UseCache>(local_embedding_ptr,
                                                           local_embedding_cache_tag_ptr,
                                                           local_embedding_cache_data_ptr,
                                                           local_rank_indice,
                                                           local_embedding_stride,
                                                           cache_set_coverage);
    per_element_ptr = optimizer_get_ptr_from_cache<UseCache>(per_element_local_embedding_ptr,
                                                             per_element_local_cache_tag_ptr,
                                                             per_element_local_cache_data_ptr,
                                                             local_rank_indice,
                                                             local_embedding_stride * 1,
                                                             cache_set_coverage);
    if (threadIdx.x == 0) {
      s_embedding_ptr   = embedding_ptr;
      s_per_element_ptr = per_element_ptr;
    }
  }
  __syncthreads();
  embedding_ptr   = s_embedding_ptr;
  per_element_ptr = s_per_element_ptr;

  float* state_sum_ptr = per_element_ptr;

  int loop_start_idx = 0;
  for (; loop_start_idx < embedding_dim; loop_start_idx += blockDim.x) {
    int local_dim_idx = threadIdx.x;
    float grad_value  = 0.0f;
    int embedding_idx = local_dim_idx + loop_start_idx;
    if (embedding_idx >= embedding_dim) { break; }
    grad_value                   = grads_ptr[embedding_idx];
    float embedding_value        = embedding_ptr[embedding_idx];
    grad_value                   = grad_value + weight_decay * embedding_value;
    float state_sum              = state_sum_ptr[embedding_idx];
    state_sum                    = state_sum + grad_value * grad_value;
    embedding_value              = embedding_value - lr * grad_value / (sqrtf(state_sum) + epsilon);
    state_sum_ptr[embedding_idx] = state_sum;
    embedding_ptr[embedding_idx] = embedding_value;
  }
}

template <typename IndiceT>
void ada_grad_optimizer_step_temp_func(const void* indices_ptr,
                                       const float* grads_ptr,
                                       float* local_embedding_ptr,
                                       uint16_t* local_embedding_cache_tag_ptr,
                                       float* local_embedding_cache_data_ptr,
                                       float* per_element_local_embedding_ptr,
                                       uint16_t* per_element_local_cache_tag_ptr,
                                       float* per_element_local_cache_data_ptr,
                                       int64_t local_entry_offset,
                                       int indice_count,
                                       int embedding_dim,
                                       int grad_stride,
                                       int local_embedding_stride,
                                       int cache_set_coverage,
                                       float weight_decay,
                                       float epsilon,
                                       float lr,
                                       cudaStream_t stream)
{
  const IndiceT* typed_indices_ptr = static_cast<const IndiceT*>(indices_ptr);
  int block_count                  = indice_count;
  if (block_count == 0) return;
  int thread_count = wholememory::div_rounding_up_unsafe(embedding_dim, 4);
  if (thread_count > 512) thread_count = 512;
  if (thread_count < 32) thread_count = 32;
  auto func_ptr = ada_grad_optimizer_step_kernel<IndiceT, false>;
  if (cache_set_coverage > 0) { func_ptr = ada_grad_optimizer_step_kernel<IndiceT, true>; }
  func_ptr<<<block_count, thread_count, 0, stream>>>(typed_indices_ptr,
                                                     grads_ptr,
                                                     local_embedding_ptr,
                                                     local_embedding_cache_tag_ptr,
                                                     local_embedding_cache_data_ptr,
                                                     per_element_local_embedding_ptr,
                                                     per_element_local_cache_tag_ptr,
                                                     per_element_local_cache_data_ptr,
                                                     local_entry_offset,
                                                     embedding_dim,
                                                     grad_stride,
                                                     local_embedding_stride,
                                                     cache_set_coverage,
                                                     weight_decay,
                                                     epsilon,
                                                     lr);
  WM_CUDA_CHECK(cudaGetLastError());
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
}

REGISTER_DISPATCH_ONE_TYPE(AdaGradOptimizerStepTempFunc,
                           ada_grad_optimizer_step_temp_func,
                           SINT3264)

wholememory_error_code_t ada_grad_optimizer_step(wholememory_tensor_t indices,
                                                 wholememory_tensor_t grads,
                                                 wholememory_tensor_t local_embedding,
                                                 wholememory_tensor_t local_embedding_cache_tag,
                                                 wholememory_tensor_t local_embedding_cache_data,
                                                 wholememory_tensor_t per_element_local_state,
                                                 wholememory_tensor_t per_element_local_cache_tag,
                                                 wholememory_tensor_t per_element_local_cache_data,
                                                 int64_t local_entry_offset,
                                                 int cache_set_coverage,
                                                 float weight_decay,
                                                 float epsilon,
                                                 float lr,
                                                 cudaStream_t stream)
{
  try {
    check_optimizer_inputs<1>(indices,
                              grads,
                              local_embedding,
                              local_embedding_cache_tag,
                              local_embedding_cache_data,
                              per_element_local_state,
                              per_element_local_cache_tag,
                              per_element_local_cache_data,
                              cache_set_coverage);
    auto* indice_desc          = wholememory_tensor_get_tensor_description(indices);
    auto* grads_desc           = wholememory_tensor_get_tensor_description(grads);
    auto* local_embedding_desc = wholememory_tensor_get_tensor_description(local_embedding);

    uint16_t* local_embedding_cache_tag_pr    = nullptr;
    float* local_embedding_cache_data_ptr     = nullptr;
    uint16_t* per_element_local_cache_tag_ptr = nullptr;
    float* per_element_local_cache_data_ptr   = nullptr;
    if (cache_set_coverage > 0) {
      local_embedding_cache_tag_pr =
        static_cast<uint16_t*>(wholememory_tensor_get_data_pointer(local_embedding_cache_tag));
      local_embedding_cache_data_ptr =
        static_cast<float*>(wholememory_tensor_get_data_pointer(local_embedding_cache_data));
      per_element_local_cache_tag_ptr =
        static_cast<uint16_t*>(wholememory_tensor_get_data_pointer(per_element_local_cache_tag));
      per_element_local_cache_data_ptr =
        static_cast<float*>(wholememory_tensor_get_data_pointer(per_element_local_cache_data));
    }

    DISPATCH_ONE_TYPE(
      indice_desc->dtype,
      AdaGradOptimizerStepTempFunc,
      wholememory_tensor_get_data_pointer(indices),
      static_cast<float*>(wholememory_tensor_get_data_pointer(grads)),
      static_cast<float*>(wholememory_tensor_get_data_pointer(local_embedding)),
      local_embedding_cache_tag_pr,
      local_embedding_cache_data_ptr,
      static_cast<float*>(wholememory_tensor_get_data_pointer(per_element_local_state)),
      per_element_local_cache_tag_ptr,
      per_element_local_cache_data_ptr,
      local_entry_offset,
      indice_desc->sizes[0],
      grads_desc->sizes[1],
      grads_desc->strides[0],
      local_embedding_desc->strides[0],
      cache_set_coverage,
      weight_decay,
      epsilon,
      lr,
      stream);
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("%s", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("%s", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (...) {
    WHOLEMEMORY_ERROR("File %s, line %d, Unknown error", __FILE__, __LINE__);
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

template <typename IndiceT, bool UseCache>
__global__ void rms_prop_optimizer_step_kernel(const IndiceT* indices_ptr,
                                               const float* grads_ptr,
                                               float* local_embedding_ptr,
                                               uint16_t* local_embedding_cache_tag_ptr,
                                               float* local_embedding_cache_data_ptr,
                                               float* per_element_local_embedding_ptr,
                                               uint16_t* per_element_local_cache_tag_ptr,
                                               float* per_element_local_cache_data_ptr,
                                               int64_t local_entry_offset,
                                               int embedding_dim,
                                               int grad_stride,
                                               int local_embedding_stride,
                                               int cache_set_coverage,
                                               float weight_decay,
                                               float epsilon,
                                               float alpha,
                                               float lr)
{
  int64_t block_idx = blockIdx.x;
  auto indice       = indices_ptr[block_idx];
  grads_ptr += block_idx * grad_stride;
  IndiceT local_rank_indice = indice - local_entry_offset;

  __shared__ float *s_embedding_ptr, *s_per_element_ptr;

  float *embedding_ptr, *per_element_ptr;
  if (threadIdx.x < 32) {
    embedding_ptr   = optimizer_get_ptr_from_cache<UseCache>(local_embedding_ptr,
                                                           local_embedding_cache_tag_ptr,
                                                           local_embedding_cache_data_ptr,
                                                           local_rank_indice,
                                                           local_embedding_stride,
                                                           cache_set_coverage);
    per_element_ptr = optimizer_get_ptr_from_cache<UseCache>(per_element_local_embedding_ptr,
                                                             per_element_local_cache_tag_ptr,
                                                             per_element_local_cache_data_ptr,
                                                             local_rank_indice,
                                                             local_embedding_stride * 1,
                                                             cache_set_coverage);
    if (threadIdx.x == 0) {
      s_embedding_ptr   = embedding_ptr;
      s_per_element_ptr = per_element_ptr;
    }
  }
  __syncthreads();
  embedding_ptr   = s_embedding_ptr;
  per_element_ptr = s_per_element_ptr;

  float* v_ptr = per_element_ptr;

  int loop_start_idx = 0;
  for (; loop_start_idx < embedding_dim; loop_start_idx += blockDim.x) {
    int local_dim_idx = threadIdx.x;
    float grad_value  = 0.0f;
    int embedding_idx = local_dim_idx + loop_start_idx;
    if (embedding_idx >= embedding_dim) { break; }
    grad_value                   = grads_ptr[local_dim_idx + loop_start_idx];
    float embedding_value        = embedding_ptr[embedding_idx];
    grad_value                   = grad_value + weight_decay * embedding_value;
    float v                      = v_ptr[embedding_idx];
    v                            = alpha * v + (1 - alpha) * grad_value * grad_value;
    embedding_value              = embedding_value - lr * grad_value / (sqrtf(v) + epsilon);
    v_ptr[embedding_idx]         = v;
    embedding_ptr[embedding_idx] = embedding_value;
  }
}

template <typename IndiceT>
void rms_prop_optimizer_step_temp_func(const void* indices_ptr,
                                       const float* grads_ptr,
                                       float* local_embedding_ptr,
                                       uint16_t* local_embedding_cache_tag_ptr,
                                       float* local_embedding_cache_data_ptr,
                                       float* per_element_local_embedding_ptr,
                                       uint16_t* per_element_local_cache_tag_ptr,
                                       float* per_element_local_cache_data_ptr,
                                       int64_t local_entry_offset,
                                       int indice_count,
                                       int embedding_dim,
                                       int grad_stride,
                                       int local_embedding_stride,
                                       int cache_set_coverage,
                                       float weight_decay,
                                       float epsilon,
                                       float alpha,
                                       float lr,
                                       cudaStream_t stream)
{
  const IndiceT* typed_indices_ptr = static_cast<const IndiceT*>(indices_ptr);
  int block_count                  = indice_count;
  if (block_count == 0) return;
  int thread_count = wholememory::div_rounding_up_unsafe(embedding_dim, 4);
  if (thread_count > 512) thread_count = 512;
  if (thread_count < 32) thread_count = 32;
  auto func_ptr = rms_prop_optimizer_step_kernel<IndiceT, false>;
  if (cache_set_coverage > 0) { func_ptr = rms_prop_optimizer_step_kernel<IndiceT, true>; }
  func_ptr<<<block_count, thread_count, 0, stream>>>(typed_indices_ptr,
                                                     grads_ptr,
                                                     local_embedding_ptr,
                                                     local_embedding_cache_tag_ptr,
                                                     local_embedding_cache_data_ptr,
                                                     per_element_local_embedding_ptr,
                                                     per_element_local_cache_tag_ptr,
                                                     per_element_local_cache_data_ptr,
                                                     local_entry_offset,
                                                     embedding_dim,
                                                     grad_stride,
                                                     local_embedding_stride,
                                                     cache_set_coverage,
                                                     weight_decay,
                                                     epsilon,
                                                     alpha,
                                                     lr);
  WM_CUDA_CHECK(cudaGetLastError());
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
}

REGISTER_DISPATCH_ONE_TYPE(RMSPropOptimizerStepTempFunc,
                           rms_prop_optimizer_step_temp_func,
                           SINT3264)

wholememory_error_code_t rms_prop_optimizer_step(wholememory_tensor_t indices,
                                                 wholememory_tensor_t grads,
                                                 wholememory_tensor_t local_embedding,
                                                 wholememory_tensor_t local_embedding_cache_tag,
                                                 wholememory_tensor_t local_embedding_cache_data,
                                                 wholememory_tensor_t per_element_local_state,
                                                 wholememory_tensor_t per_element_local_cache_tag,
                                                 wholememory_tensor_t per_element_local_cache_data,
                                                 int64_t local_entry_offset,
                                                 int cache_set_coverage,
                                                 float weight_decay,
                                                 float epsilon,
                                                 float alpha,
                                                 float lr,
                                                 cudaStream_t stream)
{
  try {
    check_optimizer_inputs<1>(indices,
                              grads,
                              local_embedding,
                              local_embedding_cache_tag,
                              local_embedding_cache_data,
                              per_element_local_state,
                              per_element_local_cache_tag,
                              per_element_local_cache_data,
                              cache_set_coverage);
    auto* indice_desc          = wholememory_tensor_get_tensor_description(indices);
    auto* grads_desc           = wholememory_tensor_get_tensor_description(grads);
    auto* local_embedding_desc = wholememory_tensor_get_tensor_description(local_embedding);

    uint16_t* local_embedding_cache_tag_pr    = nullptr;
    float* local_embedding_cache_data_ptr     = nullptr;
    uint16_t* per_element_local_cache_tag_ptr = nullptr;
    float* per_element_local_cache_data_ptr   = nullptr;
    if (cache_set_coverage > 0) {
      local_embedding_cache_tag_pr =
        static_cast<uint16_t*>(wholememory_tensor_get_data_pointer(local_embedding_cache_tag));
      local_embedding_cache_data_ptr =
        static_cast<float*>(wholememory_tensor_get_data_pointer(local_embedding_cache_data));
      per_element_local_cache_tag_ptr =
        static_cast<uint16_t*>(wholememory_tensor_get_data_pointer(per_element_local_cache_tag));
      per_element_local_cache_data_ptr =
        static_cast<float*>(wholememory_tensor_get_data_pointer(per_element_local_cache_data));
    }

    DISPATCH_ONE_TYPE(
      indice_desc->dtype,
      RMSPropOptimizerStepTempFunc,
      wholememory_tensor_get_data_pointer(indices),
      static_cast<float*>(wholememory_tensor_get_data_pointer(grads)),
      static_cast<float*>(wholememory_tensor_get_data_pointer(local_embedding)),
      local_embedding_cache_tag_pr,
      local_embedding_cache_data_ptr,
      static_cast<float*>(wholememory_tensor_get_data_pointer(per_element_local_state)),
      per_element_local_cache_tag_ptr,
      per_element_local_cache_data_ptr,
      local_entry_offset,
      indice_desc->sizes[0],
      grads_desc->sizes[1],
      grads_desc->strides[0],
      local_embedding_desc->strides[0],
      cache_set_coverage,
      weight_decay,
      epsilon,
      alpha,
      lr,
      stream);
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("%s", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("%s", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (...) {
    WHOLEMEMORY_ERROR("File %s, line %d, Unknown error", __FILE__, __LINE__);
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
