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
#include "embedding_cache.hpp"

#include <cmath>

#include "integer_utils.hpp"
#include "logger.hpp"
#include "memory_handle.hpp"
#include "wholememory_ops/functions/embedding_cache_func.h"

namespace wholememory {

embedding_cache_local_data::~embedding_cache_local_data()
{
  if (cache_line_tag_ != nullptr) {
    WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(cache_line_tag_) == WHOLEMEMORY_SUCCESS);
    cache_line_tag_ = nullptr;
  }
  if (cache_line_lfu_count_ != nullptr) {
    WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(cache_line_lfu_count_) ==
                              WHOLEMEMORY_SUCCESS);
    cache_line_lfu_count_ = nullptr;
  }
  if (cache_line_data_ != nullptr) {
    WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(cache_line_data_) == WHOLEMEMORY_SUCCESS);
    cache_line_data_ = nullptr;
  }
  if (access_count_ != nullptr) {
    WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(access_count_) == WHOLEMEMORY_SUCCESS);
    access_count_ = nullptr;
  }
}

embedding_cache_base::embedding_cache_base(wholememory_embedding_cache_policy_t cache_policy)
{
  cache_policy_ = cache_policy;
}

embedding_cache_base::~embedding_cache_base()
{
  if (cache_line_tag_wm_tensor_ != nullptr) {
    WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(cache_line_tag_wm_tensor_) ==
                              WHOLEMEMORY_SUCCESS);
    cache_line_tag_wm_tensor_ = nullptr;
  }
  if (cache_line_lfu_count_wm_tensor_ != nullptr) {
    WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(cache_line_lfu_count_wm_tensor_) ==
                              WHOLEMEMORY_SUCCESS);
    cache_line_lfu_count_wm_tensor_ = nullptr;
  }
  if (cache_line_data_wm_tensor_ != nullptr) {
    WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(cache_line_data_wm_tensor_) ==
                              WHOLEMEMORY_SUCCESS);
    cache_line_data_wm_tensor_ = nullptr;
  }
  if (access_count_wm_tensor_ != nullptr) {
    WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(access_count_wm_tensor_) ==
                              WHOLEMEMORY_SUCCESS);
    access_count_wm_tensor_ = nullptr;
  }
  if (cache_policy_ != nullptr) {
    WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_embedding_cache_policy(cache_policy_));
    cache_policy_ = nullptr;
  }
}

void embedding_cache_base::pad_last_dim(wholememory_matrix_description_t data_desc) noexcept
{
  matrix_description_           = data_desc;
  int64_t const embedding_count = matrix_description_.sizes[0];
  int64_t const embedding_dim   = matrix_description_.sizes[1];
  size_t const element_size     = wholememory_dtype_get_element_size(matrix_description_.dtype);
  WHOLEMEMORY_CHECK_NOTHROW(element_size != -1);
  int64_t const align_count      = kEmbeddingAlignmentInBytes / element_size;
  int64_t const embedding_stride = round_up_unsafe<int64_t>(embedding_dim, align_count);
  matrix_description_.stride     = embedding_stride;
  padded_matrix_description_     = matrix_description_;
}

wholememory_error_code_t embedding_cache_base::check_raw_tensor(
  wholememory_tensor_t raw_data_tensor) noexcept
{
  // Check all are same as requested.
  if (raw_data_tensor == nullptr) {
    WHOLEMEMORY_ERROR("raw_data_tensor is null");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (!wholememory_tensor_has_handle(raw_data_tensor)) {
    WHOLEMEMORY_ERROR("raw_data_tensor is not WholeMemory Tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto* mem_handle = wholememory_tensor_get_memory_handle(raw_data_tensor);
  if (mem_handle == nullptr) {
    WHOLEMEMORY_ERROR("raw_data_tensor WholeMemory Handle is nullptr");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (wholememory_get_memory_type(mem_handle) != raw_memory_type_ ||
      get_memory_location(mem_handle) != raw_memory_location_) {
    WHOLEMEMORY_ERROR(
      "raw_data_tensor WholeMemory type or location is not same as get_embedding_requirement");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_comm_t comm = nullptr;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(&comm, mem_handle));
  if (comm != raw_comm_) {
    WHOLEMEMORY_ERROR(
      "raw_data_tensor WholeMemory communicator is not same as get_embedding_requirement");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto* raw_desc = wholememory_tensor_get_tensor_description(raw_data_tensor);
  try {
    WHOLEMEMORY_CHECK(raw_desc->dim == 2 && raw_desc->storage_offset == 0);
    WHOLEMEMORY_CHECK(raw_desc->dtype == matrix_description_.dtype);
    WHOLEMEMORY_CHECK(raw_desc->strides[0] == matrix_description_.stride &&
                      raw_desc->strides[1] == 1);
    WHOLEMEMORY_CHECK(raw_desc->sizes[0] == matrix_description_.sizes[0] &&
                      raw_desc->sizes[1] == matrix_description_.sizes[1]);
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("check_raw_tensor failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_tensor_has_handle(raw_data_tensor)) {
    WHOLEMEMORY_ERROR("should be WholeMemory Tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto* root_tensor = wholememory_tensor_get_root(raw_data_tensor);
  auto* root_desc   = wholememory_tensor_get_tensor_description(root_tensor);
  try {
    WHOLEMEMORY_CHECK(root_desc->dim == 2 && root_desc->storage_offset == 0);
    WHOLEMEMORY_CHECK(root_desc->dtype == padded_matrix_description_.dtype);
    WHOLEMEMORY_CHECK(root_desc->strides[0] == padded_matrix_description_.stride &&
                      root_desc->strides[1] == 1);
    WHOLEMEMORY_CHECK(root_desc->sizes[0] == padded_matrix_description_.sizes[0] &&
                      root_desc->sizes[1] == padded_matrix_description_.sizes[1]);
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("check_raw_tensor failed for root tensor.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t embedding_cache_base::compute_cache_set_coverage() noexcept
{
  if (cache_policy_ == nullptr) {
    WHOLEMEMORY_ERROR("cache_policy_ not set.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  float const cache_ratio = cache_policy_->cache_ratio;
  if (cache_ratio >= 1.0F || cache_ratio <= 0.0F) {
    WHOLEMEMORY_ERROR("Invalid cache ratio %f, should be in range (0.0, 1.0).", cache_ratio);
    return WHOLEMEMORY_INVALID_VALUE;
  }
  cache_set_coverage_ = std::round(kCacheSetSize / cache_ratio);
  cache_set_coverage_ = std::min(cache_set_coverage_, kMaxCacheSetCoverage);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t embedding_cache_base::allocate(
  wholememory_tensor_t raw_data_tensor) noexcept
{
  WHOLEMEMORY_RETURN_ON_FAIL(check_raw_tensor(raw_data_tensor));
  padded_raw_tensor_    = wholememory_tensor_get_root(raw_data_tensor);
  auto* padded_raw_desc = wholememory_tensor_get_tensor_description(padded_raw_tensor_);
  WHOLEMEMORY_CHECK_NOTHROW(padded_raw_desc != nullptr);
  WHOLEMEMORY_CHECK_NOTHROW(padded_raw_desc->dim == 2);
  int64_t const padded_embedding_count = padded_embedding_count_for_cache_;
  WHOLEMEMORY_CHECK_NOTHROW(padded_embedding_count % cache_set_coverage_ == 0);
  int64_t const total_cache_set_count = padded_embedding_count / cache_set_coverage_;
  int cache_world_size                = 1;
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_communicator_get_size(&cache_world_size, cache_policy_->cache_comm));
  WHOLEMEMORY_CHECK_NOTHROW(total_cache_set_count % cache_world_size == 0);
  wholememory_tensor_description_t cache_line_meta_desc;
  cache_line_meta_desc.dim            = 2;
  cache_line_meta_desc.dtype          = WHOLEMEMORY_DT_INT16;
  cache_line_meta_desc.storage_offset = 0;
  cache_line_meta_desc.sizes[0]       = total_cache_set_count;
  cache_line_meta_desc.sizes[1] = cache_line_meta_desc.strides[0] = kCacheSetSize;
  cache_line_meta_desc.strides[1]                                 = 1;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_create_tensor(&cache_line_tag_wm_tensor_,
                                                       &cache_line_meta_desc,
                                                       cache_policy_->cache_comm,
                                                       cache_policy_->cache_memory_type,
                                                       cache_policy_->cache_memory_location));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_create_tensor(&cache_line_lfu_count_wm_tensor_,
                                                       &cache_line_meta_desc,
                                                       cache_policy_->cache_comm,
                                                       cache_policy_->cache_memory_type,
                                                       cache_policy_->cache_memory_location));
  wholememory_tensor_description_t cache_line_data_desc = cache_line_meta_desc;
  cache_line_data_desc.dtype                            = padded_raw_desc->dtype;
  cache_line_data_desc.sizes[0]                         = total_cache_set_count * kCacheSetSize;
  cache_line_data_desc.sizes[1]                         = padded_raw_desc->sizes[1];
  cache_line_data_desc.strides[0]                       = padded_raw_desc->strides[0];
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_create_tensor(&cache_line_data_wm_tensor_,
                                                       &cache_line_data_desc,
                                                       cache_policy_->cache_comm,
                                                       cache_policy_->cache_memory_type,
                                                       cache_policy_->cache_memory_location));
  wholememory_tensor_description_t access_count_desc;
  access_count_desc.dim            = 1;
  access_count_desc.storage_offset = 0;
  access_count_desc.sizes[0]       = padded_embedding_count_for_cache_;
  access_count_desc.dtype          = WHOLEMEMORY_DT_INT64;
  access_count_desc.strides[0]     = 1;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_create_tensor(&access_count_wm_tensor_,
                                                       &access_count_desc,
                                                       cache_policy_->cache_comm,
                                                       cache_policy_->cache_memory_type,
                                                       cache_policy_->cache_memory_location));

  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_tensor_map_local_tensor(cache_line_tag_wm_tensor_, &local_cache_.cache_line_tag_));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_tensor_map_local_tensor(
    cache_line_lfu_count_wm_tensor_, &local_cache_.cache_line_lfu_count_));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_tensor_map_local_tensor(cache_line_data_wm_tensor_,
                                                                 &local_cache_.cache_line_data_));
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_tensor_map_local_tensor(access_count_wm_tensor_, &local_cache_.access_count_));

  size_t const local_cache_line_count = wholememory_get_memory_element_count_from_tensor(
    wholememory_tensor_get_tensor_description(local_cache_.cache_line_tag_));
  WM_CUDA_CHECK_NO_THROW(
    cudaMemset(wholememory_tensor_get_data_pointer(local_cache_.cache_line_tag_),
               0,
               local_cache_line_count * sizeof(int16_t)));
  WM_CUDA_CHECK_NO_THROW(
    cudaMemset(wholememory_tensor_get_data_pointer(local_cache_.cache_line_lfu_count_),
               0,
               local_cache_line_count * sizeof(int16_t)));
  size_t const local_access_count_count = wholememory_get_memory_element_count_from_tensor(
    wholememory_tensor_get_tensor_description(local_cache_.access_count_));
  WM_CUDA_CHECK_NO_THROW(cudaMemset(wholememory_tensor_get_data_pointer(local_cache_.access_count_),
                                    0,
                                    local_access_count_count * sizeof(int64_t)));

  WM_CUDA_CHECK_NO_THROW(cudaDeviceSynchronize());
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_barrier(cache_policy_->cache_comm));

  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t embedding_cache_base::writeback_all_cache(cudaStream_t stream) noexcept
{
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t embedding_cache_base::drop_all_cache(cudaStream_t stream) noexcept
{
  return WHOLEMEMORY_SUCCESS;
}

device_cache_for_host::device_cache_for_host(wholememory_embedding_cache_policy_t cache_policy)
  : embedding_cache_base(cache_policy)
{
}

device_cache_for_host::~device_cache_for_host() {}

wholememory_error_code_t device_cache_for_host::get_embedding_requirement(
  wholememory_tensor_description_t* padded_desc,
  wholememory_matrix_description_t data_desc,
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location) noexcept
{
  if (cache_policy_ == nullptr) {
    WHOLEMEMORY_ERROR("No cache policy set.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (cache_policy_->cache_memory_location != WHOLEMEMORY_ML_DEVICE) {
    WHOLEMEMORY_ERROR("device_cache_for_host cache memory should be device.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (comm != cache_policy_->cache_comm) {
    WHOLEMEMORY_ERROR("device_cache_for_host cache should use the same communicator as raw data.");
    return WHOLEMEMORY_INVALID_VALUE;
  }
  if (padded_raw_tensor_ != nullptr) {
    WHOLEMEMORY_ERROR("embedding_cache already cached other embedding.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (memory_type > cache_policy_->cache_memory_type) {
    WHOLEMEMORY_ERROR("embedding memory_type should support at least cache memory_type.");
    return WHOLEMEMORY_INVALID_VALUE;
  }

  compute_cache_set_coverage();
  pad_last_dim(data_desc);

  int64_t const embedding_count = matrix_description_.sizes[0];

  int world_size = 1;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, comm));
  padded_embedding_count_for_cache_ = round_up_unsafe<int64_t>(
    embedding_count, static_cast<int64_t>(world_size) * cache_set_coverage_);
  padded_matrix_description_.sizes[0] = padded_embedding_count_for_cache_;
  wholememory_copy_matrix_desc_to_tensor(padded_desc, &padded_matrix_description_);

  raw_comm_            = comm;
  raw_memory_location_ = memory_location;
  raw_memory_type_     = memory_type;
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t device_cache_for_host::writeback_all_cache(cudaStream_t stream) noexcept
{
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_ops::writeback_cache_direct_same_comm(
    padded_raw_tensor_, &local_cache_, cache_set_coverage_, false, stream));
  WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
  wholememory_comm_t wm_comm;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(
    &wm_comm, wholememory_tensor_get_memory_handle(padded_raw_tensor_)));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_barrier(wm_comm));
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t device_cache_for_host::drop_all_cache(cudaStream_t stream) noexcept
{
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_ops::writeback_cache_direct_same_comm(
    padded_raw_tensor_, &local_cache_, cache_set_coverage_, true, stream));
  WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
  wholememory_comm_t wm_comm;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(
    &wm_comm, wholememory_tensor_get_memory_handle(padded_raw_tensor_)));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_barrier(wm_comm));
  return WHOLEMEMORY_SUCCESS;
}

local_cache_for_global::local_cache_for_global(wholememory_embedding_cache_policy_t cache_policy)
  : embedding_cache_base(cache_policy)
{
}

local_cache_for_global::~local_cache_for_global() {}

wholememory_error_code_t local_cache_for_global::get_embedding_requirement(
  wholememory_tensor_description_t* padded_desc,
  wholememory_matrix_description_t data_desc,
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location) noexcept
{
  if (cache_policy_ == nullptr) {
    WHOLEMEMORY_ERROR("No cache policy set.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (cache_policy_->cache_memory_type > WHOLEMEMORY_MT_CHUNKED) {
    WHOLEMEMORY_ERROR(
      "local_cache_for_global cache should support at least WHOLEMEMORY_MT_CHUNKED for now.");
    return WHOLEMEMORY_NOT_IMPLEMENTED;
  }
  if (padded_raw_tensor_ != nullptr) {
    WHOLEMEMORY_ERROR("embedding_cache already cached other embedding.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (cache_policy_->access_type != WHOLEMEMORY_AT_READONLY) {
    WHOLEMEMORY_ERROR("local_cache_for_global only READONLY cache supported.");
    return WHOLEMEMORY_NOT_IMPLEMENTED;
  }

  compute_cache_set_coverage();
  pad_last_dim(data_desc);

  int64_t const embedding_count = matrix_description_.sizes[0];
  int cache_world_size          = 1;
  if (cache_policy_->cache_comm != nullptr) {
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_communicator_get_size(&cache_world_size, cache_policy_->cache_comm));
  }
  padded_embedding_count_for_cache_ = round_up_unsafe<int64_t>(
    embedding_count, static_cast<int64_t>(cache_world_size) * cache_set_coverage_);

  padded_matrix_description_.sizes[0] = padded_embedding_count_for_cache_;
  wholememory_copy_matrix_desc_to_tensor(padded_desc, &padded_matrix_description_);
  raw_comm_            = comm;
  raw_memory_location_ = memory_location;
  raw_memory_type_     = memory_type;
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t local_cache_for_global::drop_all_cache(cudaStream_t stream) noexcept
{
  wholememory_tensor_t local_tag_tensor = local_cache_.cache_line_tag_;
  size_t local_cache_line_size          = wholememory_get_memory_size_from_tensor(
    wholememory_tensor_get_tensor_description(local_tag_tensor));
  WM_CUDA_CHECK_NO_THROW(cudaMemsetAsync(
    wholememory_tensor_get_data_pointer(local_tag_tensor), 0, local_cache_line_size, stream));
  WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));

  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_barrier(cache_policy_->cache_comm));
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory
