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
#include "gather_cached_func.h"

#include "cuda_macros.hpp"
#include "embedding_cache_func.cuh"
#include "error.hpp"
#include "gather_scatter_func.cuh"
#include "logger.hpp"
#include "wholememory_ops/register.hpp"

#include "wholememory/embedding_cache.hpp"

namespace wholememory_ops {

template <typename EmbedT, typename OutputT, typename IndexT>
__global__ void gather_cached_kernel(wholememory_gref_t padded_embedding_gref,
                                     int stride_in_int4,
                                     int start_embedding_idx,
                                     int embedding_size,
                                     wholememory_gref_t cache_line_tag_gref,
                                     wholememory_gref_t cached_embedding_gref,
                                     const IndexT* input_indices,
                                     OutputT* output,
                                     int output_stride,
                                     int cache_set_coverage,
                                     int64_t cache_start_gid,
                                     int64_t raw_start_gid)
{
  IndexT entry_gid       = input_indices[blockIdx.x];
  IndexT fixed_cache_gid = entry_gid - cache_start_gid;
  IndexT fixed_raw_gid   = entry_gid - raw_start_gid;
  IndexT cache_set_idx   = fixed_cache_gid / cache_set_coverage;
  int cache_set_lid      = static_cast<int>(fixed_cache_gid - cache_set_idx * cache_set_coverage);
  CacheLineInfo cache_line_info;
  wholememory::device_reference<uint16_t> cache_line_tag_dev_ref(cache_line_tag_gref);
  cache_line_info.LoadTag(&cache_line_tag_dev_ref[CacheLineInfo::kCacheSetSize * cache_set_idx]);
  int cache_line_index       = cache_line_info.KeyIndexSync(cache_set_lid);
  int4* padded_embedding_ptr = nullptr;
  __shared__ int4 s_embedding[32];
  EmbedT* s_embedding_embed_t = reinterpret_cast<EmbedT*>(&s_embedding[0]);
  wholememory::device_reference<int4> embedding_dev_ref(padded_embedding_gref);
  wholememory::device_reference<int4> cached_embedding_dev_ref(cached_embedding_gref);
  if (cache_line_index >= 0) {
    padded_embedding_ptr = &cached_embedding_dev_ref[(static_cast<int64_t>(cache_set_idx) *
                                                        CacheLineInfo::kCacheSetSize +
                                                      cache_line_index) *
                                                     stride_in_int4];
  } else {
    padded_embedding_ptr = &embedding_dev_ref[static_cast<int64_t>(fixed_raw_gid) * stride_in_int4];
  }
  constexpr int EMBED_TYPE_SIZE           = sizeof(EmbedT);
  constexpr int EMBED_TYPE_COUNT_PER_INT4 = 16 / EMBED_TYPE_SIZE;
  int start_int4_idx                      = EMBED_TYPE_SIZE * start_embedding_idx / 16;
  int start_padding                       = start_embedding_idx % EMBED_TYPE_COUNT_PER_INT4;
  int end_int4_idx          = (EMBED_TYPE_SIZE * (start_embedding_idx + embedding_size) + 15) / 16;
  int shared_start_idx      = start_padding;
  int output_start_idx      = 0;
  OutputT* output_embed_ptr = output + static_cast<int64_t>(blockIdx.x) * output_stride;
  for (; start_int4_idx * EMBED_TYPE_COUNT_PER_INT4 < start_embedding_idx + embedding_size;
       start_int4_idx += 32) {
    int const int4_idx = start_int4_idx + threadIdx.x;
    if (int4_idx < end_int4_idx) { s_embedding[threadIdx.x] = padded_embedding_ptr[int4_idx]; }
    int shared_end_idx =
      min(32 * EMBED_TYPE_COUNT_PER_INT4,
          start_embedding_idx + embedding_size - start_int4_idx * EMBED_TYPE_COUNT_PER_INT4);
    __syncthreads();
    while (output_start_idx < embedding_size && shared_start_idx < shared_end_idx) {
      if (shared_start_idx + threadIdx.x < shared_end_idx) {
        OutputT output_value =
          convert_type<EmbedT, OutputT>(s_embedding_embed_t[shared_start_idx + threadIdx.x]);
        output_embed_ptr[output_start_idx + threadIdx.x] = output_value;
      }
      int const data_count = min(32, shared_end_idx - shared_start_idx);
      output_start_idx += data_count;
      shared_start_idx += data_count;
    }
    shared_start_idx = 0;
    __syncthreads();
  }
}

template <typename EmbedT, typename OutputT, typename IndexT>
void gather_cached_temp_func(wholememory_gref_t padded_embedding_gref,
                             wholememory_matrix_description_t embedding_desc,
                             wholememory_gref_t cached_embedding_gref,
                             wholememory_matrix_description_t cached_embedding_desc,
                             wholememory_gref_t cache_line_tag_gref,
                             void* input_indices,
                             wholememory_array_description_t indices_desc,
                             void* output,
                             wholememory_matrix_description_t output_desc,
                             int cache_set_coverage,
                             int64_t cache_start_gid,
                             int64_t raw_start_gid,
                             cudaStream_t stream)
{
  int indice_count = indices_desc.size;
  if (indice_count == 0) return;
  WHOLEMEMORY_CHECK_NOTHROW(embedding_desc.stride == cached_embedding_desc.stride);
  WHOLEMEMORY_CHECK_NOTHROW(embedding_desc.stride *
                              wholememory_dtype_get_element_size(embedding_desc.dtype) %
                              sizeof(int4) ==
                            0);
  WHOLEMEMORY_CHECK_NOTHROW(embedding_desc.sizes[1] == output_desc.sizes[1]);
  WHOLEMEMORY_CHECK_NOTHROW(indices_desc.size == output_desc.sizes[0]);
  int stride_in_int4 =
    embedding_desc.stride * wholememory_dtype_get_element_size(embedding_desc.dtype) / sizeof(int4);
  int start_embedding_idx = embedding_desc.storage_offset;
  int embedding_size      = embedding_desc.sizes[1];
  int output_stride       = output_desc.stride;
  gather_cached_kernel<EmbedT, OutputT, IndexT><<<indice_count, 32, 0, stream>>>(
    padded_embedding_gref,
    stride_in_int4,
    start_embedding_idx,
    embedding_size,
    cache_line_tag_gref,
    cached_embedding_gref,
    static_cast<const IndexT*>(input_indices) + indices_desc.storage_offset,
    static_cast<OutputT*>(output) + output_desc.storage_offset,
    output_stride,
    cache_set_coverage,
    cache_start_gid,
    raw_start_gid);
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
}

REGISTER_DISPATCH_THREE_TYPES(
  GatherCachedFuncFloating, gather_cached_temp_func, ALLFLOAT, ALLFLOAT, SINT3264)
REGISTER_DISPATCH_THREE_TYPES(
  GatherCachedFuncInteger, gather_cached_temp_func, ALLSINT, ALLSINT, SINT3264)

wholememory_error_code_t gather_cached_func(wholememory_gref_t padded_embedding_gref,
                                            wholememory_tensor_description_t* embedding_desc,
                                            wholememory_gref_t cached_embedding_gref,
                                            wholememory_tensor_description_t* cached_embedding_desc,
                                            wholememory_gref_t cache_line_tag_gref,
                                            void* indices,
                                            wholememory_tensor_description_t* indices_desc,
                                            void* output,
                                            wholememory_tensor_description_t* output_desc,
                                            int cache_set_coverage,
                                            int64_t cache_start_gid,
                                            int64_t raw_start_gid,
                                            cudaStream_t stream)
{
  if (embedding_desc->dim != 2 || cached_embedding_desc->dim != 2 || indices_desc->dim != 1 ||
      output_desc->dim != 2) {
    WHOLEMEMORY_ERROR("tensor dim not right.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (embedding_desc->strides[1] != 1 || cached_embedding_desc->strides[1] != 1 ||
      indices_desc->strides[0] != 1 || output_desc->strides[1] != 1) {
    WHOLEMEMORY_ERROR("tensor stride of last dim should be 1.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (embedding_desc->strides[0] != cached_embedding_desc->strides[0]) {
    WHOLEMEMORY_ERROR("padded_embedding and cached_embedding should have same strides[0].");
    return WHOLEMEMORY_INVALID_VALUE;
  }
  if (embedding_desc->strides[0] * wholememory_dtype_get_element_size(embedding_desc->dtype) %
        sizeof(int4) !=
      0) {
    WHOLEMEMORY_ERROR("embedding should be aligned to int4 (16 bytes).");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (embedding_desc->sizes[1] != output_desc->sizes[1]) {
    WHOLEMEMORY_ERROR("embedding size for embedding and output should be same, %ld v.s. %ld.",
                      embedding_desc->sizes[1],
                      output_desc->sizes[1]);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (indices_desc->dtype != WHOLEMEMORY_DT_INT64 && indices_desc->dtype != WHOLEMEMORY_DT_INT) {
    WHOLEMEMORY_ERROR("indices should be int64 or int32.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (indices_desc->sizes[0] != output_desc->sizes[0]) {
    WHOLEMEMORY_ERROR("indices size and output entry count should be the same.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (embedding_desc->dtype != cached_embedding_desc->dtype) {
    WHOLEMEMORY_ERROR("embedding and cache should be same type");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_dtype_t const embedding_dtype = embedding_desc->dtype;
  wholememory_dtype_t const output_dtype    = output_desc->dtype;
  if (wholememory_dtype_is_floating_number(embedding_dtype) &&
        !wholememory_dtype_is_floating_number(output_dtype) ||
      wholememory_dtype_is_integer_number(embedding_dtype) &&
        !wholememory_dtype_is_integer_number(output_dtype)) {
    WHOLEMEMORY_ERROR("embedding and output should be all float or all integer");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_matrix_description_t embedding_matrix_desc, cached_embedding_matrix_desc,
    output_matrix_desc;
  WHOLEMEMORY_CHECK_NOTHROW(
    wholememory_convert_tensor_desc_to_matrix(&embedding_matrix_desc, embedding_desc));
  WHOLEMEMORY_CHECK_NOTHROW(wholememory_convert_tensor_desc_to_matrix(&cached_embedding_matrix_desc,
                                                                      cached_embedding_desc));
  WHOLEMEMORY_CHECK_NOTHROW(
    wholememory_convert_tensor_desc_to_matrix(&output_matrix_desc, output_desc));
  wholememory_array_description_t indices_array_desc;
  WHOLEMEMORY_CHECK_NOTHROW(
    wholememory_convert_tensor_desc_to_array(&indices_array_desc, indices_desc));
  if (indices_array_desc.size == 0) return WHOLEMEMORY_SUCCESS;

  if (wholememory_dtype_is_floating_number(embedding_dtype)) {
    DISPATCH_THREE_TYPES(embedding_dtype,
                         output_dtype,
                         indices_desc->dtype,
                         GatherCachedFuncFloating,
                         padded_embedding_gref,
                         embedding_matrix_desc,
                         cached_embedding_gref,
                         cached_embedding_matrix_desc,
                         cache_line_tag_gref,
                         indices,
                         indices_array_desc,
                         output,
                         output_matrix_desc,
                         cache_set_coverage,
                         cache_start_gid,
                         raw_start_gid,
                         stream);
  } else {
    DISPATCH_THREE_TYPES(embedding_dtype,
                         output_dtype,
                         indices_desc->dtype,
                         GatherCachedFuncInteger,
                         padded_embedding_gref,
                         embedding_matrix_desc,
                         cached_embedding_gref,
                         cached_embedding_matrix_desc,
                         cache_line_tag_gref,
                         indices,
                         indices_array_desc,
                         output,
                         output_matrix_desc,
                         cache_set_coverage,
                         cache_start_gid,
                         raw_start_gid,
                         stream);
  }
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
  return WHOLEMEMORY_SUCCESS;
}

template <typename EmbedT, typename OutputT, typename IndexT>
__global__ void try_gather_cached_kernel(int stride_in_int4,
                                         int start_embedding_idx,
                                         int embedding_size,
                                         wholememory_gref_t cache_line_tag_gref,
                                         wholememory_gref_t cached_embedding_gref,
                                         const IndexT* input_indices,
                                         IndexT* hit_indices,
                                         IndexT* miss_indices,
                                         OutputT* output,
                                         int output_stride,
                                         int cache_set_coverage,
                                         int64_t cache_start_gid)
{
  IndexT entry_gid       = input_indices[blockIdx.x];
  IndexT fixed_cache_gid = entry_gid - cache_start_gid;
  IndexT cache_set_idx   = fixed_cache_gid / cache_set_coverage;
  int cache_set_lid      = static_cast<int>(fixed_cache_gid - cache_set_idx * cache_set_coverage);
  CacheLineInfo cache_line_info;
  wholememory::device_reference<uint16_t> cache_line_tag_dev_ref(cache_line_tag_gref);
  cache_line_info.LoadTag(&cache_line_tag_dev_ref[CacheLineInfo::kCacheSetSize * cache_set_idx]);
  int cache_line_index       = cache_line_info.KeyIndexSync(cache_set_lid);
  int4* padded_embedding_ptr = nullptr;
  __shared__ int4 s_embedding[32];
  EmbedT* s_embedding_embed_t = reinterpret_cast<EmbedT*>(&s_embedding[0]);
  wholememory::device_reference<int4> cached_embedding_dev_ref(cached_embedding_gref);
  if (cache_line_index >= 0) {
    padded_embedding_ptr =
      &cached_embedding_dev_ref[static_cast<int64_t>(cache_set_idx * CacheLineInfo::kCacheSetSize +
                                                     cache_line_index) *
                                stride_in_int4];
  }
  if (threadIdx.x == 0) {
    if (hit_indices) hit_indices[blockIdx.x] = cache_line_index >= 0 ? entry_gid : (IndexT)-1;
    if (miss_indices) miss_indices[blockIdx.x] = cache_line_index >= 0 ? (IndexT)-1 : entry_gid;
  }
  if (cache_line_index < 0) return;
  constexpr int EMBED_TYPE_SIZE           = sizeof(EmbedT);
  constexpr int EMBED_TYPE_COUNT_PER_INT4 = 16 / EMBED_TYPE_SIZE;
  int start_int4_idx                      = EMBED_TYPE_SIZE * start_embedding_idx / 16;
  int start_padding                       = start_embedding_idx % EMBED_TYPE_COUNT_PER_INT4;
  int end_int4_idx          = (EMBED_TYPE_SIZE * (start_embedding_idx + embedding_size) + 15) / 16;
  int shared_start_idx      = start_padding;
  int output_start_idx      = 0;
  OutputT* output_embed_ptr = output + static_cast<int64_t>(blockIdx.x) * output_stride;
  for (; start_int4_idx * EMBED_TYPE_COUNT_PER_INT4 < start_embedding_idx + embedding_size;
       start_int4_idx += 32) {
    int const int4_idx = start_int4_idx + threadIdx.x;
    if (int4_idx < end_int4_idx) { s_embedding[threadIdx.x] = padded_embedding_ptr[int4_idx]; }
    int shared_end_idx =
      min(32 * EMBED_TYPE_COUNT_PER_INT4,
          start_embedding_idx + embedding_size - start_int4_idx * EMBED_TYPE_COUNT_PER_INT4);
    __syncthreads();
    while (output_start_idx < embedding_size && shared_start_idx < shared_end_idx) {
      if (shared_start_idx + threadIdx.x < shared_end_idx) {
        OutputT output_value =
          convert_type<EmbedT, OutputT>(s_embedding_embed_t[shared_start_idx + threadIdx.x]);
        output_embed_ptr[output_start_idx + threadIdx.x] = output_value;
      }
      int const data_count = min(32, shared_end_idx - shared_start_idx);
      output_start_idx += data_count;
      shared_start_idx += data_count;
    }
    shared_start_idx = 0;
    __syncthreads();
  }
}

template <typename EmbedT, typename OutputT, typename IndexT>
void try_gather_cached_temp_func(wholememory_gref_t cached_embedding_gref,
                                 wholememory_matrix_description_t cached_embedding_desc,
                                 wholememory_gref_t cache_line_tag_gref,
                                 void* input_indices,
                                 wholememory_array_description_t indices_desc,
                                 void* hit_indices,
                                 void* miss_indices,
                                 void* output,
                                 wholememory_matrix_description_t output_desc,
                                 int cache_set_coverage,
                                 int64_t cache_start_gid,
                                 cudaStream_t stream)
{
  int indice_count = indices_desc.size;
  WHOLEMEMORY_CHECK_NOTHROW(cached_embedding_desc.stride *
                              wholememory_dtype_get_element_size(cached_embedding_desc.dtype) %
                              sizeof(int4) ==
                            0);
  WHOLEMEMORY_CHECK_NOTHROW(cached_embedding_desc.sizes[1] == output_desc.sizes[1]);
  WHOLEMEMORY_CHECK_NOTHROW(indices_desc.size == output_desc.sizes[0]);
  int stride_in_int4 = cached_embedding_desc.stride *
                       wholememory_dtype_get_element_size(cached_embedding_desc.dtype) /
                       sizeof(int4);
  int start_embedding_idx = cached_embedding_desc.storage_offset;
  int embedding_size      = cached_embedding_desc.sizes[1];
  int output_stride       = output_desc.stride;
  try_gather_cached_kernel<EmbedT, OutputT, IndexT><<<indice_count, 32, 0, stream>>>(
    stride_in_int4,
    start_embedding_idx,
    embedding_size,
    cache_line_tag_gref,
    cached_embedding_gref,
    static_cast<const IndexT*>(input_indices) + indices_desc.storage_offset,
    static_cast<IndexT*>(hit_indices),
    static_cast<IndexT*>(miss_indices),
    static_cast<OutputT*>(output) + output_desc.storage_offset,
    output_stride,
    cache_set_coverage,
    cache_start_gid);
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
}

REGISTER_DISPATCH_THREE_TYPES(
  TryGatherCachedFuncFloating, try_gather_cached_temp_func, ALLFLOAT, ALLFLOAT, SINT3264)
REGISTER_DISPATCH_THREE_TYPES(
  TryGatherCachedFuncInteger, try_gather_cached_temp_func, ALLSINT, ALLSINT, SINT3264)

wholememory_error_code_t try_gather_cached_func(
  wholememory_gref_t cached_embedding_gref,
  wholememory_tensor_description_t* cached_embedding_desc,
  wholememory_gref_t cache_line_tag_gref,
  void* indices,
  wholememory_tensor_description_t* indices_desc,
  void* hit_indices,
  void* miss_indices,
  void* output,
  wholememory_tensor_description_t* output_desc,
  int cache_set_coverage,
  int64_t cache_start_gid,
  cudaStream_t stream)
{
  if (cached_embedding_desc->dim != 2 || indices_desc->dim != 1 || output_desc->dim != 2) {
    WHOLEMEMORY_ERROR("tensor dim not right.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (cached_embedding_desc->strides[1] != 1 || indices_desc->strides[0] != 1 ||
      output_desc->strides[1] != 1) {
    WHOLEMEMORY_ERROR("tensor stride of last dim should be 1.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (cached_embedding_desc->strides[0] *
        wholememory_dtype_get_element_size(cached_embedding_desc->dtype) % sizeof(int4) !=
      0) {
    WHOLEMEMORY_ERROR("cached_embedding_desc should be aligned to int4 (16 bytes).");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (cached_embedding_desc->sizes[1] != output_desc->sizes[1]) {
    WHOLEMEMORY_ERROR("embedding size for embedding and output should be same.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (indices_desc->dtype != WHOLEMEMORY_DT_INT64 && indices_desc->dtype != WHOLEMEMORY_DT_INT) {
    WHOLEMEMORY_ERROR("indices should be int64 or int32.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (indices_desc->sizes[0] != output_desc->sizes[0]) {
    WHOLEMEMORY_ERROR("indices size and output entry count should be the same.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_dtype_t const embedding_dtype = cached_embedding_desc->dtype;
  wholememory_dtype_t const output_dtype    = output_desc->dtype;
  if (wholememory_dtype_is_floating_number(embedding_dtype) &&
        !wholememory_dtype_is_floating_number(output_dtype) ||
      wholememory_dtype_is_integer_number(embedding_dtype) &&
        !wholememory_dtype_is_integer_number(output_dtype)) {
    WHOLEMEMORY_ERROR("embedding and output should be all float or all integer");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_matrix_description_t embedding_matrix_desc, cached_embedding_matrix_desc,
    output_matrix_desc;
  WHOLEMEMORY_CHECK_NOTHROW(
    wholememory_convert_tensor_desc_to_matrix(&embedding_matrix_desc, cached_embedding_desc));
  WHOLEMEMORY_CHECK_NOTHROW(wholememory_convert_tensor_desc_to_matrix(&cached_embedding_matrix_desc,
                                                                      cached_embedding_desc));
  WHOLEMEMORY_CHECK_NOTHROW(
    wholememory_convert_tensor_desc_to_matrix(&output_matrix_desc, output_desc));
  wholememory_array_description_t indices_array_desc;
  WHOLEMEMORY_CHECK_NOTHROW(
    wholememory_convert_tensor_desc_to_array(&indices_array_desc, indices_desc));

  if (indices_array_desc.size == 0) return WHOLEMEMORY_SUCCESS;

  if (wholememory_dtype_is_floating_number(embedding_dtype)) {
    DISPATCH_THREE_TYPES(embedding_dtype,
                         output_dtype,
                         indices_desc->dtype,
                         TryGatherCachedFuncFloating,
                         cached_embedding_gref,
                         cached_embedding_matrix_desc,
                         cache_line_tag_gref,
                         indices,
                         indices_array_desc,
                         hit_indices,
                         miss_indices,
                         output,
                         output_matrix_desc,
                         cache_set_coverage,
                         cache_start_gid,
                         stream);
  } else {
    DISPATCH_THREE_TYPES(embedding_dtype,
                         output_dtype,
                         indices_desc->dtype,
                         TryGatherCachedFuncInteger,
                         cached_embedding_gref,
                         cached_embedding_matrix_desc,
                         cache_line_tag_gref,
                         indices,
                         indices_array_desc,
                         hit_indices,
                         miss_indices,
                         output,
                         output_matrix_desc,
                         cache_set_coverage,
                         cache_start_gid,
                         stream);
  }
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
