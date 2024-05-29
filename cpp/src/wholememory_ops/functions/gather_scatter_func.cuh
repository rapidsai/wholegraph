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

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <wholememory/device_reference.cuh>
#include <wholememory/global_reference.h>
#include <wholememory/tensor_description.h>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "wholememory/integer_utils.hpp"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace wholememory_ops {

template <typename DataTypeT>
__device__ __forceinline__ void mov_typed_data(DataTypeT* to, const DataTypeT* from)
{
  *to = *from;
}
template <int DATA_SIZE>
__device__ __forceinline__ void mov_data(void* to, const void* from)
{
  char* ptr_to         = static_cast<char*>(to);
  const char* ptr_from = static_cast<const char*>(from);
  for (int i = 0; i < DATA_SIZE; i++) {
    ptr_to[i] = ptr_from[i];
  }
}
template <typename DataTypeT, int DATA_SIZE>
struct typed_data_vector {
  DataTypeT data[DATA_SIZE];
};
template <>
struct typed_data_vector<double, 2> {
  double2 data;
};
template <>
struct typed_data_vector<int64_t, 2> {
  int4 data;
};
template <>
struct typed_data_vector<float, 2> {
  float2 data;
};
template <>
struct typed_data_vector<float, 4> {
  float4 data;
};
template <>
struct typed_data_vector<int, 2> {
  int2 data;
};
template <>
struct typed_data_vector<int, 4> {
  int4 data;
};
template <>
struct typed_data_vector<__half, 2> {
  __half2 data;
};
template <>
struct typed_data_vector<__half, 4> {
  int2 data;
};
template <>
struct typed_data_vector<__half, 8> {
  int4 data;
};
template <>
struct typed_data_vector<int16_t, 2> {
  int data;
};
template <>
struct typed_data_vector<int16_t, 4> {
  int2 data;
};
template <>
struct typed_data_vector<int16_t, 8> {
  int4 data;
};
template <>
struct typed_data_vector<nv_bfloat16, 2> {
  nv_bfloat162 data;
};
template <>
struct typed_data_vector<nv_bfloat16, 4> {
  int2 data;
};
template <>
struct typed_data_vector<nv_bfloat16, 8> {
  int4 data;
};
template <>
struct typed_data_vector<int8_t, 2> {
  int16_t data;
};
template <>
struct typed_data_vector<int8_t, 4> {
  int data;
};
template <>
struct typed_data_vector<int8_t, 8> {
  int2 data;
};
template <>
struct typed_data_vector<int8_t, 16> {
  int4 data;
};
template <typename DataTypeT, int DATA_SIZE>
__device__ __forceinline__ DataTypeT& typed_data_vector_at(
  typed_data_vector<DataTypeT, DATA_SIZE>& v, int idx)
{
  return ((DataTypeT*)(&v.data))[idx];
}

template <>
__device__ __forceinline__ void mov_data<1>(void* to, const void* from)
{
  mov_typed_data(static_cast<int8_t*>(to), static_cast<const int8_t*>(from));
}
template <>
__device__ __forceinline__ void mov_data<2>(void* to, const void* from)
{
  mov_typed_data(static_cast<int16_t*>(to), static_cast<const int16_t*>(from));
}
template <>
__device__ __forceinline__ void mov_data<4>(void* to, const void* from)
{
  mov_typed_data(static_cast<int32_t*>(to), static_cast<const int32_t*>(from));
}
template <>
__device__ __forceinline__ void mov_data<8>(void* to, const void* from)
{
  mov_typed_data(static_cast<int64_t*>(to), static_cast<const int64_t*>(from));
}
template <>
__device__ __forceinline__ void mov_data<16>(void* to, const void* from)
{
  mov_typed_data(static_cast<int4*>(to), static_cast<const int4*>(from));
}

template <typename DataTypeT>
class type_caster {
 public:
  using LoadTypeT  = DataTypeT;
  using StoreTypeT = DataTypeT;
  static __device__ __forceinline__ LoadTypeT convert_load_data(DataTypeT data)
  {
    return static_cast<LoadTypeT>(data);
  }
  static __device__ __forceinline__ DataTypeT convert_store_data(StoreTypeT data)
  {
    return static_cast<DataTypeT>(data);
  }
};
template <>
class type_caster<__half> {
 public:
  using LoadTypeT  = float;
  using StoreTypeT = float;
  static __device__ __forceinline__ LoadTypeT convert_load_data(__half data)
  {
    return static_cast<LoadTypeT>(data);
  }
  static __device__ __forceinline__ __half convert_store_data(StoreTypeT data)
  {
    return static_cast<__half>(data);
  }
};
template <>
class type_caster<__nv_bfloat16> {
 public:
  using LoadTypeT  = float;
  using StoreTypeT = float;
  static __device__ LoadTypeT convert_load_data(__nv_bfloat16 data)
  {
    return static_cast<LoadTypeT>(data);
  }
  static __device__ __nv_bfloat16 convert_store_data(StoreTypeT data)
  {
    return static_cast<__nv_bfloat16>(data);
  }
};

template <typename FromT, typename ToT>
__device__ __forceinline__ ToT convert_type(FromT from)
{
  return type_caster<ToT>::convert_store_data(type_caster<FromT>::convert_load_data(from));
}

/**
 * Determine alignment of a WholeMemory matrix, in element count, maximum 16 / element_size.
 * @param embedding_desc : wholememory_matrix_description_t matrix description.
 * @return : Alignment that can be used, in element count.
 */
inline int determine_wholememory_alignment_elt_count(
  wholememory_matrix_description_t embedding_desc)
{
  int elt_size = static_cast<int>(wholememory_dtype_get_element_size(embedding_desc.dtype));
  WHOLEMEMORY_CHECK(elt_size != -1);
  int alignment = 16 / elt_size;
  for (; alignment > 1; alignment /= 2) {
    if (embedding_desc.storage_offset % alignment == 0 &&
        embedding_desc.sizes[1] % alignment == 0 && embedding_desc.stride % alignment == 0)
      break;
  }
  return alignment;
}

/**
 * Determine alignment of normal memory, in element count, maximum 16 / element_size.
 * @param ptr : pointer to the memory.
 * @param memory_desc : wholememory_matrix_description_t matrix description.
 * @return : Alignment that can be used, in element count.
 */
inline int determine_memory_alignment_elt_count(const void* ptr,
                                                wholememory_matrix_description_t memory_desc)
{
  int elt_size = static_cast<int>(wholememory_dtype_get_element_size(memory_desc.dtype));
  WHOLEMEMORY_CHECK(elt_size != -1);
  int alignment   = 16 / elt_size;
  int64_t int_ptr = reinterpret_cast<int64_t>(ptr);
  WHOLEMEMORY_CHECK(int_ptr % elt_size == 0);
  int_ptr /= elt_size;
  int_ptr += memory_desc.storage_offset;
  for (; alignment > 1; alignment /= 2) {
    if (int_ptr % alignment == 0 && memory_desc.sizes[1] % alignment == 0 &&
        memory_desc.stride % alignment == 0)
      break;
  }
  return alignment;
}

template <typename EmbeddingT, typename IndexT, typename OutputT, int ALIGNMENT = 1>
__global__ void gather_func_kernel(wholememory_gref_t embedding_gref,
                                   wholememory_matrix_description_t embedding_desc,
                                   const IndexT* indices,
                                   int64_t indice_count,
                                   bool gather_with_sorted_ids,
                                   const IndexT* raw_indices,
                                   OutputT* output,
                                   wholememory_matrix_description_t output_desc)
{
  auto block  = cooperative_groups::this_thread_block();
  auto mywarp = cooperative_groups::tiled_partition<32>(block);
  __shared__ char shm_in_char[16384];
  OutputT* all_sh = reinterpret_cast<OutputT*>(shm_in_char);
  OutputT* my_shared;
  int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
  int lane_id = threadIdx.x % 32;

  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t output_stride    = output_desc.stride;
  int shm_size             = 16384 / sizeof(OutputT);
  wholememory::device_reference<EmbeddingT> embedding_dev_ref(embedding_gref);

  typed_data_vector<EmbeddingT, ALIGNMENT> embeddings;
  typed_data_vector<OutputT, ALIGNMENT> outputs;

  bool use_shm = true;
  if (shm_size / (blockDim.x / 32) < output_desc.sizes[1]) {  //
    use_shm = false;
  } else {
    my_shared = all_sh + shm_size / (blockDim.x / 32) * (threadIdx.x / 32);
  }

  for (int64_t output_idx = warp_id; output_idx < indice_count;
       output_idx += gridDim.x * (blockDim.x / 32)) {
    int64_t raw_output_idx =
      gather_with_sorted_ids ? (int64_t)(raw_indices[output_idx]) : output_idx;
    OutputT* output_ptr = output + output_desc.storage_offset + output_stride * raw_output_idx;
    if (!use_shm) { my_shared = output_ptr; }
    int64_t embedding_table_idx = indices[output_idx];
    if (embedding_table_idx < 0) continue;
    EmbeddingT* emb_ptr =
      &embedding_dev_ref[embedding_desc.storage_offset + embedding_table_idx * embedding_stride];

    for (int emb_idx = lane_id * ALIGNMENT; emb_idx < embedding_size; emb_idx += ALIGNMENT * 32) {
      mov_data<sizeof(EmbeddingT) * ALIGNMENT>(&embeddings, emb_ptr + emb_idx);
#pragma unroll
      for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
        typed_data_vector_at(outputs, sub_idx) =
          convert_type<EmbeddingT, OutputT>(typed_data_vector_at(embeddings, sub_idx));
      }
      mov_data<sizeof(OutputT) * ALIGNMENT>(my_shared + emb_idx, &outputs);
    }
    if (use_shm) {
      int copy_size = output_desc.sizes[1] * sizeof(OutputT);
      cooperative_groups::memcpy_async(mywarp, output_ptr, my_shared, copy_size);
      cooperative_groups::wait(mywarp);
    }
  }
  return;
}

template <int N>
struct IsPowerOfTwo {
  static constexpr bool value = (N > 0) && ((N & (N - 1)) == 0);
};

template <typename EmbeddingT,
          typename IndexT,
          typename OutputT,
          int SUB_WARP_SIZE = 1,
          int ALIGNMENT     = 1>
__global__ void gather_func_sub_warp_kernel(wholememory_gref_t embedding_gref,
                                            wholememory_matrix_description_t embedding_desc,
                                            const IndexT* indices,
                                            int64_t indice_count,
                                            bool gather_with_sorted_ids,
                                            const IndexT* raw_indices,
                                            OutputT* output,
                                            wholememory_matrix_description_t output_desc)
{
  static_assert(IsPowerOfTwo<SUB_WARP_SIZE>::value && SUB_WARP_SIZE < 32,
                "SUB_WARP_SIZE must be the power of 2,and smaller than 32.");

  auto block = cooperative_groups::this_thread_block();

  auto subwarp     = cooperative_groups::tiled_partition<SUB_WARP_SIZE>(block);
  int sub_warp_id  = subwarp.meta_group_size() * blockIdx.x + subwarp.meta_group_rank();
  int sub_warp_num = subwarp.meta_group_size() * gridDim.x;

  int lane_id_in_sub_warp = subwarp.thread_rank();
  wholememory::device_reference<EmbeddingT> embedding_dev_ref(embedding_gref);

  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t output_stride    = output_desc.stride;

  typed_data_vector<EmbeddingT, ALIGNMENT> embeddings;
  typed_data_vector<OutputT, ALIGNMENT> outputs;
  for (int64_t output_idx = sub_warp_id; output_idx < indice_count; output_idx += sub_warp_num) {
    int64_t raw_output_idx =
      gather_with_sorted_ids ? (int64_t)(raw_indices[output_idx]) : output_idx;
    OutputT* output_ptr = output + output_desc.storage_offset + output_stride * raw_output_idx;
    IndexT embedding_table_idx = indices[output_idx];
    if (embedding_table_idx < 0) continue;
    int64_t embedding_offset =
      embedding_desc.storage_offset + embedding_table_idx * embedding_stride;

    for (int emb_idx = lane_id_in_sub_warp * ALIGNMENT; emb_idx < embedding_size;
         emb_idx += ALIGNMENT * SUB_WARP_SIZE) {
      mov_data<sizeof(EmbeddingT) * ALIGNMENT>(&embeddings,
                                               &embedding_dev_ref[embedding_offset + emb_idx]);
#pragma unroll
      for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
        typed_data_vector_at(outputs, sub_idx) =
          convert_type<EmbeddingT, OutputT>(typed_data_vector_at(embeddings, sub_idx));
      }
      mov_data<sizeof(OutputT) * ALIGNMENT>(output_ptr + emb_idx, &outputs);
    }
  }
}

template <typename EmbeddingT, typename IndexT, typename OutputT>
void gather_temp_func(wholememory_gref_t embedding_gref,
                      wholememory_matrix_description_t embedding_desc,
                      void* indices,
                      int64_t indice_count,
                      bool gather_with_sorted_ids,
                      void* raw_indices,
                      void* output,
                      wholememory_matrix_description_t output_desc,
                      cudaStream_t stream,
                      int gather_sms)
{
  WHOLEMEMORY_EXPECTS(output_desc.sizes[0] == indice_count,
                      "gather_func, output shape[0]=%ld, but indice_count=%ld",
                      output_desc.sizes[0],
                      indice_count);
  if (indice_count == 0 || embedding_desc.sizes[1] == 0) return;
  int wm_alignment = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment = determine_memory_alignment_elt_count(output, output_desc);
  int alignment    = std::min<int>(wm_alignment, mm_alignment);
  // int embedding_size = embedding_desc.sizes[1];
  // int thread_num       = wholememory::div_rounding_up_safe<int>(embedding_size, alignment);
  // thread_num           = std::min(thread_num, 512);
  // int64_t block_count = indice_count >= 1024 ? 1024 : static_cast<int>(indice_count);

  void (*kernel_fn)(wholememory_gref_t,
                    wholememory_matrix_description_t,
                    const IndexT*,
                    int64_t,
                    bool,
                    const IndexT*,
                    OutputT*,
                    wholememory_matrix_description_t) = nullptr;

  switch (alignment) {
    case 16: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 16>;
      break;
    }
    case 8: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 8>;
      break;
    }
    case 4: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 4>;
      break;
    }
    case 2: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 2>;
      break;
    }
    case 1: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 1>;
      break;
    }
    default: {
      WHOLEMEMORY_FAIL("gather func alignment=%d.", alignment);
      return;
    }
  }
  int block_size  = 1024;
  int block_count = indice_count > 1568 ? 1568 : indice_count;
  if (gather_sms != -1) block_count = gather_sms;

  // for small embedding size ,use subwarp to gather
  int min_threads_per_embedding = embedding_desc.sizes[1] / alignment;
  if (min_threads_per_embedding < 32) {
#define SWITCH_GATHER_FUNC_WITH_ALIGNMENT(KERNEL_NAME, SUB_WARP_SIZE)          \
  switch (alignment) {                                                         \
    case 16: {                                                                 \
      kernel_fn = KERNEL_NAME<EmbeddingT, IndexT, OutputT, SUB_WARP_SIZE, 16>; \
      break;                                                                   \
    }                                                                          \
    case 8: {                                                                  \
      kernel_fn = KERNEL_NAME<EmbeddingT, IndexT, OutputT, SUB_WARP_SIZE, 8>;  \
      break;                                                                   \
    }                                                                          \
    case 4: {                                                                  \
      kernel_fn = KERNEL_NAME<EmbeddingT, IndexT, OutputT, SUB_WARP_SIZE, 4>;  \
      break;                                                                   \
    }                                                                          \
    case 2: {                                                                  \
      kernel_fn = KERNEL_NAME<EmbeddingT, IndexT, OutputT, SUB_WARP_SIZE, 2>;  \
      break;                                                                   \
    }                                                                          \
    case 1: {                                                                  \
      kernel_fn = KERNEL_NAME<EmbeddingT, IndexT, OutputT, SUB_WARP_SIZE, 1>;  \
      break;                                                                   \
    }                                                                          \
    default: {                                                                 \
      WHOLEMEMORY_FAIL("gather func alignment=%d.", alignment);                \
      return;                                                                  \
    }                                                                          \
  }

    int threads_per_embedding = 16;
    if (min_threads_per_embedding >= 16) {
      SWITCH_GATHER_FUNC_WITH_ALIGNMENT(gather_func_sub_warp_kernel, 16);
      threads_per_embedding = 16;
    } else if (min_threads_per_embedding < 16 && min_threads_per_embedding >= 8) {
      SWITCH_GATHER_FUNC_WITH_ALIGNMENT(gather_func_sub_warp_kernel, 8);
      threads_per_embedding = 8;
    } else if (min_threads_per_embedding < 8 && min_threads_per_embedding >= 4) {
      SWITCH_GATHER_FUNC_WITH_ALIGNMENT(gather_func_sub_warp_kernel, 4);
      threads_per_embedding = 4;
    } else if (min_threads_per_embedding < 4 && min_threads_per_embedding >= 2) {
      SWITCH_GATHER_FUNC_WITH_ALIGNMENT(gather_func_sub_warp_kernel, 2);
      threads_per_embedding = 2;
    } else {
      SWITCH_GATHER_FUNC_WITH_ALIGNMENT(gather_func_sub_warp_kernel, 1);
      threads_per_embedding = 1;
    }

#undef SWITCH_GATHER_FUNC_WITH_ALIGNMENT
    block_size            = 128;
    int max_blocks_per_sm = 8;
    WM_CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel_fn, block_size, 0));

    int sm_count  = 100;
    int device_id = 0;
    WM_CUDA_CHECK(cudaGetDevice(&device_id));
    WM_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));

    // block_count = indice_count > 1568 ? 1568 : indice_count;
    int min_embedding_per_block = block_size / threads_per_embedding;
    block_count = min((int)(indice_count + min_embedding_per_block - 1) / min_embedding_per_block,
                      sm_count * max_blocks_per_sm * 4);
    if (gather_sms != -1) block_count = gather_sms * max_blocks_per_sm;
  }
  kernel_fn<<<block_count, block_size, 0, stream>>>(embedding_gref,
                                                    embedding_desc,
                                                    static_cast<const IndexT*>(indices),
                                                    indice_count,
                                                    gather_with_sorted_ids,
                                                    static_cast<const IndexT*>(raw_indices),
                                                    static_cast<OutputT*>(output),
                                                    output_desc);
  WM_CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename IndexT, typename EmbeddingT, int ALIGNMENT = 1>
__global__ void scatter_func_kernel(const InputT* input,
                                    wholememory_matrix_description_t input_desc,
                                    const IndexT* indices,
                                    int64_t indice_count,
                                    wholememory_gref_t embedding_gref,
                                    wholememory_matrix_description_t embedding_desc)
{
  auto block  = cooperative_groups::this_thread_block();
  auto mywarp = cooperative_groups::tiled_partition<32>(block);
  __shared__ char shm_in_char[24576];
  InputT* all_sh = reinterpret_cast<InputT*>(shm_in_char);
  InputT* my_shared;
  int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
  int lane_id = threadIdx.x % 32;

  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t input_stride     = input_desc.stride;
  int async_copy_align     = sizeof(InputT) > 4 ? 1 : 4 / sizeof(InputT);

  int shm_size = 24576 / sizeof(InputT);

  int batch_size = (shm_size / (blockDim.x / 32) - async_copy_align) /
                   input_stride;  // indices batch size in lines
  wholememory::device_reference<EmbeddingT> embedding_dev_ref(embedding_gref);

  typed_data_vector<EmbeddingT, ALIGNMENT> embeddings;
  typed_data_vector<InputT, ALIGNMENT> inputs;
  int input_off_tail =
    input_desc.storage_offset %
    async_copy_align;  // this is crutial for copy alignment, 4 bytes as alignment;
  bool use_shm = true;
  if (batch_size <= 0) {
    use_shm    = false;
    batch_size = 1;
  } else {
    my_shared = all_sh + shm_size / (blockDim.x / 32) * (threadIdx.x / 32);
  }
  for (int64_t input_idx = warp_id * batch_size; input_idx < indice_count;
       input_idx += gridDim.x * (blockDim.x / 32) * batch_size) {
    int cur_idx_lines =
      (indice_count - input_idx) > batch_size ? batch_size : indice_count - input_idx;
    const InputT* input_ptr =
      input + input_desc.storage_offset - input_off_tail + input_stride * input_idx;
    // this variable is also for alignment
    if (use_shm) {
      int copy_size = input_off_tail + cur_idx_lines * input_stride;
      if (input_idx + cur_idx_lines < indice_count)  // input_dim * sizeof(InputT) > 4 is needed
        copy_size = (copy_size + async_copy_align - 1) / async_copy_align * async_copy_align;
      copy_size *= sizeof(InputT);
      cooperative_groups::memcpy_async(mywarp, my_shared, input_ptr, copy_size);
      cooperative_groups::wait(mywarp);
    }
    for (int e = 0; e < cur_idx_lines; e++) {
      int64_t embedding_table_idx = indices[input_idx + e];
      if (embedding_table_idx < 0) continue;
      EmbeddingT* emb_ptr =
        &embedding_dev_ref[embedding_desc.storage_offset + embedding_table_idx * embedding_stride];

      for (int emb_idx = lane_id * ALIGNMENT; emb_idx < embedding_size; emb_idx += ALIGNMENT * 32) {
        if (use_shm)
          mov_data<sizeof(InputT) * ALIGNMENT>(
            &inputs, my_shared + input_off_tail + e * input_stride + emb_idx);
        else
          mov_data<sizeof(InputT) * ALIGNMENT>(
            &inputs, input_ptr + input_off_tail + e * input_stride + emb_idx);
#pragma unroll
        for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
          typed_data_vector_at(embeddings, sub_idx) =
            convert_type<InputT, EmbeddingT>(typed_data_vector_at(inputs, sub_idx));
        }
        mov_data<sizeof(EmbeddingT) * ALIGNMENT>(emb_ptr + emb_idx, &embeddings);
      }
    }
    mywarp.sync();
  }
  return;
}

template <typename InputT, typename IndexT, typename EmbeddingT>
void scatter_temp_func(const void* input,
                       wholememory_matrix_description_t input_desc,
                       void* indices,
                       int64_t indice_count,
                       wholememory_gref_t embedding_gref,
                       wholememory_matrix_description_t embedding_desc,
                       cudaStream_t stream,
                       int scatter_sms)
{
  WHOLEMEMORY_EXPECTS(input_desc.sizes[0] == indice_count,
                      "scatter_func, input shape[0]=%ld, but indice_count=%ld",
                      input_desc.sizes[0],
                      indice_count);
  if (indice_count == 0 || embedding_desc.sizes[1] == 0) return;
  int wm_alignment = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment = determine_memory_alignment_elt_count(input, input_desc);
  int alignment    = std::min<int>(wm_alignment, mm_alignment);

  void (*kernel_fn)(const InputT*,
                    wholememory_matrix_description_t,
                    const IndexT*,
                    int64_t,
                    wholememory_gref_t,
                    wholememory_matrix_description_t) = nullptr;
  switch (alignment) {
    case 16: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 16>;
      break;
    }
    case 8: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 8>;
      break;
    }
    case 4: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 4>;
      break;
    }
    case 2: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 2>;
      break;
    }
    case 1: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 1>;
      break;
    }
    default: {
      WHOLEMEMORY_FAIL("scatter func alignment=%d.", alignment);
      return;
    }
  }
  int block_size  = 256;
  int block_count = indice_count > 1568 ? 1568 : indice_count;
  if (scatter_sms != -1) block_count = scatter_sms;
  kernel_fn<<<block_count, block_size, 0, stream>>>(static_cast<const InputT*>(input),
                                                    input_desc,
                                                    static_cast<const IndexT*>(indices),
                                                    indice_count,
                                                    embedding_gref,
                                                    embedding_desc);
  WM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace wholememory_ops
