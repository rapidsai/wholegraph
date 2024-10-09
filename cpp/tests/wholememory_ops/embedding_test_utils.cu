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
#include "embedding_test_utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <stdio.h>

#include <experimental/random>

#include <wholememory_ops/register.hpp>
#include <wholememory_ops/temp_memory_handle.hpp>

namespace wholememory_ops {
namespace testing {

template <typename DataTypeT>
class type_convertor {
 public:
  using LoadTypeT  = DataTypeT;
  using StoreTypeT = DataTypeT;
  static __device__ LoadTypeT convert_load_data(DataTypeT data)
  {
    return static_cast<LoadTypeT>(data);
  }
  static __device__ DataTypeT convert_store_data(StoreTypeT data)
  {
    return static_cast<DataTypeT>(data);
  }
};
template <>
class type_convertor<__half> {
 public:
  using LoadTypeT  = float;
  using StoreTypeT = float;
  static __device__ LoadTypeT convert_load_data(__half data)
  {
    return static_cast<LoadTypeT>(data);
  }
  static __device__ __half convert_store_data(StoreTypeT data) { return static_cast<__half>(data); }
};
template <>
class type_convertor<__nv_bfloat16> {
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

template <typename SrcTypeT, typename DstTypeT>
__global__ void matrix_type_cast_kernel(DstTypeT* dst,
                                        const SrcTypeT* src,
                                        int64_t row_count,
                                        int64_t col_count,
                                        int dst_stride,
                                        int src_stride)
{
  int row_count_per_block = blockDim.x / col_count;
  int row_idx_in_block    = threadIdx.x / col_count;
  int col_idx             = threadIdx.x - row_idx_in_block * col_count;
  int64_t row_idx = static_cast<int64_t>(blockIdx.x) * row_count_per_block + row_idx_in_block;
  if (row_idx_in_block >= row_count_per_block || row_idx >= row_count) return;
  auto src_data        = src[row_idx * src_stride + col_idx];
  auto loaded_src_data = type_convertor<SrcTypeT>::convert_load_data(src_data);
  auto store_dst_data = static_cast<typename type_convertor<DstTypeT>::StoreTypeT>(loaded_src_data);
  auto dst_data       = type_convertor<DstTypeT>::convert_store_data(store_dst_data);
  dst[row_idx * dst_stride + col_idx] = dst_data;
}

template <typename SrcTypeT, typename DstTypeT>
void matrix_test_cast(void* dst,
                      const void* src,
                      int64_t row_count,
                      int64_t col_count,
                      int dst_stride,
                      int src_stride,
                      cudaStream_t stream)
{
  int threads_per_block = std::max<int>(col_count, 256);
  int rows_per_block    = threads_per_block / col_count;
  threads_per_block     = rows_per_block * col_count;
  int block_count       = static_cast<int>((row_count + rows_per_block - 1) / rows_per_block);
  matrix_type_cast_kernel<SrcTypeT, DstTypeT>
    <<<block_count, threads_per_block, 0, stream>>>(static_cast<DstTypeT*>(dst),
                                                    static_cast<const SrcTypeT*>(src),
                                                    row_count,
                                                    col_count,
                                                    dst_stride,
                                                    src_stride);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

REGISTER_DISPATCH_TWO_TYPES(FloatMatrixTestCast,
                            matrix_test_cast,
                            HALF_FLOAT_DOUBLE,
                            HALF_FLOAT_DOUBLE)
REGISTER_DISPATCH_TWO_TYPES(IntMatrixTestCast, matrix_test_cast, ALLSINT, ALLSINT)

void device_matrix_type_cast(void* dst,
                             wholememory_matrix_description_t dst_desc,
                             const void* src,
                             wholememory_matrix_description_t src_desc,
                             cudaStream_t stream)
{
  EXPECT_EQ(dst_desc.sizes[0], src_desc.sizes[0]);
  EXPECT_EQ(dst_desc.sizes[1], src_desc.sizes[1]);
  bool is_float_src =
    src_desc.dtype == WHOLEMEMORY_DT_HALF || src_desc.dtype == WHOLEMEMORY_DT_FLOAT ||
    src_desc.dtype == WHOLEMEMORY_DT_DOUBLE || src_desc.dtype == WHOLEMEMORY_DT_BF16;
  bool is_float_dst =
    dst_desc.dtype == WHOLEMEMORY_DT_HALF || dst_desc.dtype == WHOLEMEMORY_DT_FLOAT ||
    dst_desc.dtype == WHOLEMEMORY_DT_DOUBLE || dst_desc.dtype == WHOLEMEMORY_DT_BF16;
  EXPECT_EQ(is_float_dst, is_float_src);
  if (is_float_src) {
    DISPATCH_TWO_TYPES(src_desc.dtype,
                       dst_desc.dtype,
                       FloatMatrixTestCast,
                       dst,
                       src,
                       src_desc.sizes[0],
                       src_desc.sizes[1],
                       dst_desc.stride,
                       src_desc.stride,
                       stream);
  } else {
    DISPATCH_TWO_TYPES(src_desc.dtype,
                       dst_desc.dtype,
                       IntMatrixTestCast,
                       dst,
                       src,
                       src_desc.sizes[0],
                       src_desc.sizes[1],
                       dst_desc.stride,
                       src_desc.stride,
                       stream);
  }
}

void device_array_type_cast(void* dst,
                            wholememory_array_description_t dst_desc,
                            const void* src,
                            wholememory_array_description_t src_desc,
                            cudaStream_t stream)
{
  wholememory_tensor_description_t dst_tensor_desc, src_tensor_desc;
  wholememory_copy_array_desc_to_tensor(&dst_tensor_desc, &dst_desc);
  wholememory_copy_array_desc_to_tensor(&src_tensor_desc, &src_desc);
  wholememory_matrix_description_t dst_matrix_desc, src_matrix_desc;
  EXPECT_TRUE(wholememory_convert_tensor_desc_to_matrix(&dst_matrix_desc, &dst_tensor_desc));
  EXPECT_TRUE(wholememory_convert_tensor_desc_to_matrix(&src_matrix_desc, &src_tensor_desc));
  device_matrix_type_cast(dst, dst_matrix_desc, src, src_matrix_desc, stream);
}

template <typename DataTypeT>
__device__ __forceinline__ DataTypeT device_get_data_from_int64(int64_t data)
{
  return static_cast<DataTypeT>(data);
}

template <typename DataTypeT>
__device__ __forceinline__ DataTypeT device_cast_int64_t_to_float(int64_t data)
{
  return static_cast<DataTypeT>(data);
}
template <>
__device__ __forceinline__ __half device_cast_int64_t_to_float(int64_t data)
{
  return static_cast<__half>(static_cast<float>(data));
}
template <>
__device__ __forceinline__ __nv_bfloat16 device_cast_int64_t_to_float(int64_t data)
{
  return static_cast<__nv_bfloat16>(static_cast<float>(data));
}

template <int M, int E, typename DataTypeT>
__device__ __forceinline__ DataTypeT device_get_float_data_from_int64(int64_t data)
{
  static_assert(M > 0, "M should be larget than 0.");
  static_assert(E > 0, "M should be larget than 0.");
  static_assert(M + E + 1 == sizeof(DataTypeT) * 8, "M + E should be sizeof(DataTypeT) * 8 - 1");
  int64_t mdata   = data & ((1LL << (M + 1)) - 1LL);
  auto data_float = device_cast_int64_t_to_float<DataTypeT>(mdata);
  return data_float;
}

template <>
__device__ __forceinline__ float device_get_data_from_int64<float>(int64_t data)
{
  return device_get_float_data_from_int64<23, 8, float>(data);
}
template <>
__device__ __forceinline__ double device_get_data_from_int64<double>(int64_t data)
{
  return device_get_float_data_from_int64<52, 11, double>(data);
}
template <>
__device__ __forceinline__ __half device_get_data_from_int64<__half>(int64_t data)
{
  return device_get_float_data_from_int64<10, 5, __half>(data);
}
template <>
__device__ __forceinline__ __nv_bfloat16 device_get_data_from_int64<__nv_bfloat16>(int64_t data)
{
  return device_get_float_data_from_int64<7, 8, __nv_bfloat16>(data);
}

template <typename DataTypeT>
__device__ __forceinline__ DataTypeT device_get_embedding_data(int64_t embedding_idx,
                                                               int embedding_dim,
                                                               int dim_idx)
{
  int64_t embedding_data = embedding_idx * embedding_dim + dim_idx;
  embedding_data         = embedding_data * 97 + 1007;
  embedding_data         = embedding_idx;
  return device_get_data_from_int64<DataTypeT>(embedding_data);
}

template <typename DataTypeT>
__global__ void get_embedding_data_kernel(DataTypeT* embedding_ptr,
                                          int64_t storage_offset,
                                          int embedding_dim,
                                          int embedding_stride,
                                          int64_t local_entry_start,
                                          int64_t local_entry_count)
{
  int64_t local_embedding_idx = blockIdx.x;
  if (local_embedding_idx >= local_entry_count) return;
  int thread_x = threadIdx.x;
  embedding_ptr += storage_offset;
  int64_t embedding_idx = local_entry_start + local_embedding_idx;
  embedding_ptr += embedding_stride * local_embedding_idx;
  for (; thread_x < embedding_dim; thread_x += blockDim.x) {
    auto data = device_get_embedding_data<DataTypeT>(embedding_idx, embedding_dim, thread_x);
    embedding_ptr[thread_x] = data;
  }
}

template <typename DataTypeT>
void get_embedding_data(void* embedding_ptr,
                        wholememory_matrix_description_t embedding_desc,
                        int64_t local_entry_start,
                        int64_t local_entry_count,
                        cudaStream_t stream)
{
  int64_t storage_offset = embedding_desc.storage_offset;
  int embedding_dim      = embedding_desc.sizes[1];
  int embedding_stride   = embedding_desc.stride;
  int block_size         = embedding_dim;
  block_size             = std::min(block_size, 256);
  int block_count        = local_entry_count;
  get_embedding_data_kernel<DataTypeT>
    <<<block_count, block_size, 0, stream>>>(static_cast<DataTypeT*>(embedding_ptr),
                                             storage_offset,
                                             embedding_dim,
                                             embedding_stride,
                                             local_entry_start,
                                             local_entry_count);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
}

void device_random_init_local_embedding_table(wholememory_handle_t embedding_handle,
                                              wholememory_matrix_description_t embedding_desc,
                                              cudaStream_t stream)
{
  void* local_embedding_ptr;
  size_t local_embedding_size, local_embedding_offset;

  EXPECT_EQ(
    wholememory_get_local_memory(
      &local_embedding_ptr, &local_embedding_size, &local_embedding_offset, embedding_handle),
    WHOLEMEMORY_SUCCESS);

  int64_t embedding_entry_size =
    embedding_desc.stride * wholememory_dtype_get_element_size(embedding_desc.dtype);

  EXPECT_EQ(local_embedding_size % embedding_entry_size, 0);
  EXPECT_EQ(local_embedding_offset % embedding_entry_size, 0);

  int64_t local_entry_start = local_embedding_offset / embedding_entry_size;
  int64_t local_entry_count = local_embedding_size / embedding_entry_size;

  if (local_entry_count == 0) return;

  switch (embedding_desc.dtype) {
    case WHOLEMEMORY_DT_FLOAT: {
      get_embedding_data<float>(
        local_embedding_ptr, embedding_desc, local_entry_start, local_entry_count, stream);
      break;
    }
    case WHOLEMEMORY_DT_HALF: {
      get_embedding_data<__half>(
        local_embedding_ptr, embedding_desc, local_entry_start, local_entry_count, stream);
      break;
    }
    case WHOLEMEMORY_DT_DOUBLE: {
      get_embedding_data<double>(
        local_embedding_ptr, embedding_desc, local_entry_start, local_entry_count, stream);
      break;
    }
    case WHOLEMEMORY_DT_BF16: {
      get_embedding_data<__nv_bfloat16>(
        local_embedding_ptr, embedding_desc, local_entry_start, local_entry_count, stream);
      break;
    }
    case WHOLEMEMORY_DT_INT: {
      get_embedding_data<int>(
        local_embedding_ptr, embedding_desc, local_entry_start, local_entry_count, stream);
      break;
    }
    case WHOLEMEMORY_DT_INT64: {
      get_embedding_data<int64_t>(
        local_embedding_ptr, embedding_desc, local_entry_start, local_entry_count, stream);
      break;
    }
    case WHOLEMEMORY_DT_INT16: {
      get_embedding_data<int16_t>(
        local_embedding_ptr, embedding_desc, local_entry_start, local_entry_count, stream);
      break;
    }
    case WHOLEMEMORY_DT_INT8: {
      get_embedding_data<int8_t>(
        local_embedding_ptr, embedding_desc, local_entry_start, local_entry_count, stream);
      break;
    }
    default: {
      FAIL();
      break;
    }
  }
}

template <typename IndexT, typename GenTypeT>
__global__ void device_get_expected_embedding_kernel(GenTypeT* gen_buffer,
                                                     int64_t storage_offset,
                                                     int embedding_dim,
                                                     int embedding_stride,
                                                     const IndexT* indices,
                                                     int indice_count)
{
  int64_t block_idx = blockIdx.x;
  if (block_idx >= indice_count) return;
  int thread_x = threadIdx.x;
  gen_buffer += storage_offset;
  int64_t embedding_idx = indices[block_idx];
  gen_buffer += embedding_stride * block_idx;
  for (; thread_x < embedding_dim; thread_x += blockDim.x) {
    auto data = device_get_embedding_data<GenTypeT>(embedding_idx, embedding_dim, thread_x);
    gen_buffer[thread_x] = data;
  }
}

template <typename IndexT, typename GenTypeT>
void device_get_expected_embedding_temp_func(void* gen_buffer,
                                             wholememory_matrix_description_t gen_buffer_desc,
                                             void* indices,
                                             wholememory_array_description_t indices_desc,
                                             cudaStream_t stream)
{
  EXPECT_EQ(gen_buffer_desc.sizes[0], indices_desc.size);
  if (indices_desc.size == 0) return;
  int block_count  = gen_buffer_desc.sizes[0];
  int thread_count = std::min<int>(gen_buffer_desc.sizes[1], 256);
  device_get_expected_embedding_kernel<IndexT, GenTypeT>
    <<<block_count, thread_count, 0, stream>>>(static_cast<GenTypeT*>(gen_buffer),
                                               gen_buffer_desc.storage_offset,
                                               gen_buffer_desc.sizes[1],
                                               gen_buffer_desc.stride,
                                               static_cast<const IndexT*>(indices),
                                               indices_desc.size);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

REGISTER_DISPATCH_TWO_TYPES(DeviceGetExpectedEmbedding,
                            device_get_expected_embedding_temp_func,
                            SINT3264,
                            ALLSINT_ALLFLOAT)

void device_get_expected_embedding(void* output,
                                   wholememory_matrix_description_t output_desc,
                                   wholememory_dtype_t embedding_dtype,
                                   void* indices,
                                   wholememory_array_description_t indices_desc,
                                   wholememory_env_func_t* p_env_fns,
                                   cudaStream_t stream)
{
  void* gen_buffer = output;
  wholememory_ops::temp_memory_handle gen_buffer_tmh(p_env_fns);
  auto gen_desc = output_desc;
  if (embedding_dtype != output_desc.dtype) {
    gen_desc.dtype          = embedding_dtype;
    gen_desc.stride         = gen_desc.sizes[1];
    gen_desc.storage_offset = 0;
    gen_buffer              = gen_buffer_tmh.device_malloc(
      wholememory_get_memory_element_count_from_matrix(&gen_desc), gen_desc.dtype);
  }
  DISPATCH_TWO_TYPES(indices_desc.dtype,
                     gen_desc.dtype,
                     DeviceGetExpectedEmbedding,
                     gen_buffer,
                     gen_desc,
                     indices,
                     indices_desc,
                     stream);
  if (embedding_dtype != output_desc.dtype) {
    device_matrix_type_cast(output, output_desc, gen_buffer, gen_desc, stream);
  }
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
}

template <typename IndexT>
void host_get_random_indices(void* indices,
                             wholememory_array_description_t indice_desc,
                             int64_t max_indices)
{
  IndexT* indices_ptr = static_cast<IndexT*>(indices);
  std::experimental::reseed();
  for (int64_t i = 0; i < indice_desc.size; i++) {
    IndexT random_index = std::experimental::randint<IndexT>(0, max_indices - 1);
    indices_ptr[i + indice_desc.storage_offset] = random_index;
  }
}

void host_random_init_indices(void* indices,
                              wholememory_array_description_t indices_desc,
                              int64_t max_indices)
{
  EXPECT_TRUE(indices_desc.dtype == WHOLEMEMORY_DT_INT ||
              indices_desc.dtype == WHOLEMEMORY_DT_INT64);
  if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_random_indices<int>(indices, indices_desc, max_indices);
  } else {
    host_get_random_indices<int64_t>(indices, indices_desc, max_indices);
  }
}

template <typename DataTypeT>
uint64_t load_hex_data(void* ptr, size_t offset)
{
  DataTypeT* data_ptr = static_cast<DataTypeT*>(ptr) + offset;
  uint64_t data       = *data_ptr;
  return data;
}

void host_check_embedding_same(void* host_embedding,
                               wholememory_matrix_description_t embedding_desc,
                               void* host_reference,
                               wholememory_matrix_description_t reference_desc)
{
  EXPECT_EQ(embedding_desc.dtype, reference_desc.dtype);
  EXPECT_EQ(embedding_desc.sizes[0], reference_desc.sizes[0]);
  EXPECT_EQ(embedding_desc.sizes[1], reference_desc.sizes[1]);
  int64_t row_count   = embedding_desc.sizes[0];
  int64_t col_count   = embedding_desc.sizes[1];
  size_t element_size = wholememory_dtype_get_element_size(embedding_desc.dtype);
  int64_t diff_count  = 0;
  for (int64_t row = 0; row < row_count; row++) {
    for (int64_t col = 0; col < col_count; col++) {
      uint64_t embedding_data, reference_data;
      if (element_size == 1) {
        embedding_data = load_hex_data<uint8_t>(
          host_embedding, row * embedding_desc.stride + embedding_desc.storage_offset + col);
        reference_data = load_hex_data<uint8_t>(
          host_reference, row * reference_desc.stride + reference_desc.storage_offset + col);
      } else if (element_size == 2) {
        embedding_data = load_hex_data<uint16_t>(
          host_embedding, row * embedding_desc.stride + embedding_desc.storage_offset + col);
        reference_data = load_hex_data<uint16_t>(
          host_reference, row * reference_desc.stride + reference_desc.storage_offset + col);
      } else if (element_size == 4) {
        embedding_data = load_hex_data<uint32_t>(
          host_embedding, row * embedding_desc.stride + embedding_desc.storage_offset + col);
        reference_data = load_hex_data<uint32_t>(
          host_reference, row * reference_desc.stride + reference_desc.storage_offset + col);
      } else {
        embedding_data = load_hex_data<uint64_t>(
          host_embedding, row * embedding_desc.stride + embedding_desc.storage_offset + col);
        reference_data = load_hex_data<uint64_t>(
          host_reference, row * reference_desc.stride + reference_desc.storage_offset + col);
      }
      if (embedding_data != reference_data) {
        if (diff_count < 10) {
          printf("row=%ld, col=%ld, got %lx (float %f), but should be %lx (float %f)\n",
                 row,
                 col,
                 embedding_data,
                 *(float*)(&embedding_data),
                 reference_data,
                 *(float*)(&reference_data));
          fflush(stdout);
          EXPECT_EQ(embedding_data, reference_data);
        }
        diff_count++;
      }
    }
  }
  EXPECT_EQ(diff_count, 0);
}

void host_random_init_float(float* data, int64_t len, float max_value, float min_value)
{
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(-1.0, 1.0);  // rage 0 - 1
  for (int64_t i = 0; i < len; i++) {
    data[i] = dis(e);
  }
}

void host_random_partition(size_t* partition_sizes, size_t total_size, int partition_count)
{
  std::default_random_engine random_engine(0);
  std::uniform_int_distribution<size_t> uniform(90, 100);
  size_t acc_size   = 0;
  size_t random_sum = 0;
  for (int i = 0; i < partition_count; i++) {
    partition_sizes[i] = (size_t)uniform(random_engine);
    random_sum += partition_sizes[i];
  }
  for (int i = 0; i < partition_count; i++) {
    partition_sizes[i] = (size_t)((partition_sizes[i] / (double)random_sum) * total_size);
    acc_size += partition_sizes[i];
  }
  partition_sizes[0] += total_size - acc_size;
}

}  // namespace testing
}  // namespace wholememory_ops
