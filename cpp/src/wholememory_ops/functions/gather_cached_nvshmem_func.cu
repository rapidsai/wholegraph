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
#ifdef WITH_NVSHMEM_SUPPORT
#include "gather_cached_func.h"

#include "cuda_macros.hpp"
#include "embedding_cache_func.cuh"
#include "error.hpp"
#include "gather_scatter_func.cuh"
#include "logger.hpp"
#include "wholememory/device_reference.cuh"
#include "wholememory/global_reference.h"
#include "wholememory/wholememory.h"
#include "wholememory_ops/register.hpp"

#include "wholememory/embedding_cache.hpp"
#include <cstddef>
#include <cstdint>

namespace wholememory_ops {

template <typename EmbedT, typename OutputT, typename IndexT>
__global__ void gather_cached_nvshmem_global_kernel(wholememory_gref_t padded_embedding_nvshmem_ref,
                                                    int stride_in_int4,
                                                    int start_embedding_idx,
                                                    int embedding_size,
                                                    wholememory_nvshmem_ref_t cache_line_tag_gref,
                                                    wholememory_nvshmem_ref_t cached_embedding_gref,
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
  wholememory::nvshmem_device_reference<uint16_t> cache_line_tag_dev_ref{cache_line_tag_gref};
  cache_line_info.LoadTag_nvshmem(
    cache_line_tag_dev_ref.load(CacheLineInfo::kCacheSetSize * cache_set_idx + threadIdx.x));
  const int cache_line_index = cache_line_info.KeyIndexSync(cache_set_lid);
  __shared__ int4 s_embedding[32];
  EmbedT* s_embedding_embed_t = reinterpret_cast<EmbedT*>(&s_embedding[0]);
  wholememory::device_reference<int4> embedding_dev_ref(padded_embedding_nvshmem_ref);
  wholememory::nvshmem_device_reference<int4> cached_embedding_dev_ref(cached_embedding_gref);
  int64_t ref_offset = 0;
  if (cache_line_index >= 0) {
    ref_offset =
      (static_cast<int64_t>(cache_set_idx) * CacheLineInfo::kCacheSetSize + cache_line_index) *
      stride_in_int4;
  } else {
    ref_offset = static_cast<int64_t>(fixed_raw_gid) * stride_in_int4;
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
    if (int4_idx < end_int4_idx) {
      s_embedding[threadIdx.x] = cache_line_index >= 0
                                   ? cached_embedding_dev_ref.load(ref_offset + int4_idx)
                                   : embedding_dev_ref[ref_offset + int4_idx];
    }
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
__global__ void gather_cached_nvshmem_nvshemm_kernel(
  wholememory_nvshmem_ref_t padded_embedding_nvshmem_ref,
  int stride_in_int4,
  int start_embedding_idx,
  int embedding_size,
  wholememory_nvshmem_ref_t cache_line_tag_gref,
  wholememory_nvshmem_ref_t cached_embedding_gref,
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
  wholememory::nvshmem_device_reference<uint16_t> cache_line_tag_dev_ref{cache_line_tag_gref};
  cache_line_info.LoadTag_nvshmem(
    cache_line_tag_dev_ref.load(CacheLineInfo::kCacheSetSize * cache_set_idx + threadIdx.x));
  int cache_line_index = cache_line_info.KeyIndexSync(cache_set_lid);
  __shared__ int4 s_embedding[32];
  EmbedT* s_embedding_embed_t = reinterpret_cast<EmbedT*>(&s_embedding[0]);
  wholememory::nvshmem_device_reference<int4> embedding_dev_ref(padded_embedding_nvshmem_ref);
  wholememory::nvshmem_device_reference<int4> cached_embedding_dev_ref(cached_embedding_gref);
  wholememory::nvshmem_device_reference<int4> padded_embedding_dev_nv_ref{
    wholememory_nvshmem_ref_t{}};
  int64_t ref_offset = 0;
  if (cache_line_index >= 0) {
    padded_embedding_dev_nv_ref = cached_embedding_dev_ref;
    ref_offset =
      (static_cast<int64_t>(cache_set_idx) * CacheLineInfo::kCacheSetSize + cache_line_index) *
      stride_in_int4;
  } else {
    padded_embedding_dev_nv_ref = embedding_dev_ref;
    ref_offset                  = static_cast<int64_t>(fixed_raw_gid) * stride_in_int4;
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
    if (int4_idx < end_int4_idx) {
      s_embedding[threadIdx.x] = padded_embedding_dev_nv_ref.load(ref_offset + int4_idx);
    }
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
wholememory_error_code_t gather_cached_nvshmem_temp_func(
  wholememory_handle_t padded_embedding_handle,
  wholememory_matrix_description_t embedding_desc,
  wholememory_handle_t cached_embedding_handle,
  wholememory_matrix_description_t cached_embedding_desc,
  wholememory_handle_t cache_line_tag_handle,
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
  if (indice_count == 0) return WHOLEMEMORY_SUCCESS;
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

  wholememory_memory_type_t embedding_type = wholememory_get_memory_type(padded_embedding_handle);

  wholememory_nvshmem_ref_t cached_embedding_nvshmem_ref;
  wholememory_nvshmem_ref_t cache_line_tag_nvshmem_ref;
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_get_nvshmem_reference(&cached_embedding_nvshmem_ref, cached_embedding_handle));
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_get_nvshmem_reference(&cache_line_tag_nvshmem_ref, cache_line_tag_handle));

  if (embedding_type == WHOLEMEMORY_MT_DISTRIBUTED &&
      (wholememory_get_distributed_backend(padded_embedding_handle) == WHOLEMEMORY_DB_NVSHMEM)) {
    wholememory_nvshmem_ref_t padded_embedding_nvshmem_ref;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_get_nvshmem_reference(&padded_embedding_nvshmem_ref, padded_embedding_handle));
    gather_cached_nvshmem_nvshemm_kernel<EmbedT, OutputT, IndexT><<<indice_count, 32, 0, stream>>>(
      padded_embedding_nvshmem_ref,
      stride_in_int4,
      start_embedding_idx,
      embedding_size,
      cache_line_tag_nvshmem_ref,
      cached_embedding_nvshmem_ref,
      static_cast<const IndexT*>(input_indices) + indices_desc.storage_offset,
      static_cast<OutputT*>(output) + output_desc.storage_offset,
      output_stride,
      cache_set_coverage,
      cache_start_gid,
      raw_start_gid);
    WM_CUDA_DEBUG_SYNC_STREAM(stream);
  } else {
    wholememory_gref_t padded_embedding_ref;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_get_global_reference(&padded_embedding_ref, padded_embedding_handle));

    gather_cached_nvshmem_global_kernel<EmbedT, OutputT, IndexT><<<indice_count, 32, 0, stream>>>(
      padded_embedding_ref,
      stride_in_int4,
      start_embedding_idx,
      embedding_size,
      cache_line_tag_nvshmem_ref,
      cached_embedding_nvshmem_ref,
      static_cast<const IndexT*>(input_indices) + indices_desc.storage_offset,
      static_cast<OutputT*>(output) + output_desc.storage_offset,
      output_stride,
      cache_set_coverage,
      cache_start_gid,
      raw_start_gid);
    WM_CUDA_DEBUG_SYNC_STREAM(stream);
  }
  return WHOLEMEMORY_SUCCESS;
}

REGISTER_DISPATCH_THREE_TYPES(GatherCachedNvshmemFuncTemp,
                              gather_cached_nvshmem_temp_func,
                              ALLSINT_ALLFLOAT,
                              ALLSINT_ALLFLOAT,
                              SINT3264);

wholememory_error_code_t gather_cached_nvshmem_func(
  wholememory_handle_t padded_embedding_handle,
  wholememory_tensor_description_t* embedding_desc,
  wholememory_handle_t cached_embedding_handle,
  wholememory_tensor_description_t* cached_embedding_desc,
  wholememory_handle_t cache_line_tag_handle,
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

  DISPATCH_THREE_TYPES(embedding_dtype,
                       output_dtype,
                       indices_desc->dtype,
                       GatherCachedNvshmemFuncTemp,
                       padded_embedding_handle,
                       embedding_matrix_desc,
                       cached_embedding_handle,
                       cached_embedding_matrix_desc,
                       cache_line_tag_handle,
                       indices,
                       indices_array_desc,
                       output,
                       output_matrix_desc,
                       cache_set_coverage,
                       cache_start_gid,
                       raw_start_gid,
                       stream);

  WM_CUDA_DEBUG_SYNC_STREAM(stream);
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops

#endif
