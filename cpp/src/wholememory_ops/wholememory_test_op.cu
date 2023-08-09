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
#include <wholememory/wholememory_op.h>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"
#include "output_memory_handle.hpp"
#include "register.hpp"
#include "temp_memory_handle.hpp"

template <typename DataTypeT>
__global__ void EnvTestTempFUnc(const DataTypeT* input_ptr,
                                DataTypeT* output_ptr,
                                int64_t emb_dim,
                                int64_t output_stride)
{
  int id = blockIdx.x;
  output_ptr += output_stride * id;
  float f_id = static_cast<float>(id);
  for (int idx = threadIdx.x; idx < emb_dim; idx += blockDim.x) {
    output_ptr[idx] = static_cast<DataTypeT>(f_id) + input_ptr[idx];
  }
}

template <typename DataTypeT>
void EnvTestTempFunc(const void* input_ptr,
                     void* out_ptr,
                     int64_t emb_dim,
                     int64_t entry_count,
                     int64_t output_stride,
                     cudaStream_t stream)
{
  int thread_count = std::min<int>(emb_dim, 512);
  int block_count  = entry_count;
  EnvTestTempFUnc<DataTypeT>
    <<<block_count, thread_count, 0, stream>>>(static_cast<const DataTypeT*>(input_ptr),
                                               static_cast<DataTypeT*>(out_ptr),
                                               emb_dim,
                                               output_stride);
  WM_CUDA_CHECK_NO_THROW(cudaGetLastError());
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
}

REGISTER_DISPATCH_ONE_TYPE(EnvTestTempFunc, EnvTestTempFunc, ALLSINT_ALLFLOAT)

#ifdef __cplusplus
extern "C" {
#endif

wholememory_error_code_t wholememory_env_test_op(wholememory_tensor_t input_tensor,
                                                 wholememory_tensor_t output_fixed_tensor,
                                                 void* output_variable_device_tensor_handle,
                                                 void* output_variable_pinned_tensor_handle,
                                                 void* output_variable_host_tensor_handle,
                                                 int64_t output_variable_entry_count,
                                                 wholememory_env_func_t* p_env_fns,
                                                 void* stream)
{
  auto* input_desc  = wholememory_tensor_get_tensor_description(input_tensor);
  auto* output_desc = wholememory_tensor_get_tensor_description(output_fixed_tensor);
  WHOLEMEMORY_CHECK_NOTHROW(input_desc->dim == 1);
  int64_t emb_dim = input_desc->sizes[0];
  WHOLEMEMORY_CHECK_NOTHROW(output_desc->dim == 2);
  WHOLEMEMORY_CHECK_NOTHROW(output_desc->sizes[0] == output_variable_entry_count);
  WHOLEMEMORY_CHECK_NOTHROW(output_desc->sizes[1] == emb_dim);
  WHOLEMEMORY_CHECK_NOTHROW(input_desc->dtype == output_desc->dtype);

  wholememory_ops::output_memory_handle out_device_handle(p_env_fns,
                                                          output_variable_device_tensor_handle);
  wholememory_ops::output_memory_handle out_pinned_handle(p_env_fns,
                                                          output_variable_pinned_tensor_handle);
  wholememory_ops::output_memory_handle out_host_handle(p_env_fns,
                                                        output_variable_host_tensor_handle);

  wholememory_ops::temp_memory_handle temp_buffer_handle(p_env_fns);
  // fprintf(stderr, "===> IN OP start allocate temp device ptr.\n");
  void* temp_buffer_ptr =
    temp_buffer_handle.device_malloc(output_variable_entry_count * emb_dim, input_desc->dtype);
  // fprintf(stderr, "===> IN OP temp device allocated=%ld\n",
  // reinterpret_cast<int64_t>(temp_buffer_ptr));
  size_t output_size =
    output_variable_entry_count * emb_dim * wholememory_dtype_get_element_size(input_desc->dtype);

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

  // fprintf(stderr, "===> IN OP start computing.\n");
  DISPATCH_ONE_TYPE(input_desc->dtype,
                    EnvTestTempFunc,
                    wholememory_tensor_get_data_pointer(input_tensor),
                    temp_buffer_ptr,
                    emb_dim,
                    output_variable_entry_count,
                    output_desc->strides[0],
                    cuda_stream);

  // fprintf(stderr, "===> IN OP compute done.\n");

  // fprintf(stderr, "===> IN OP start allocate output device ptr.\n");
  void* output_device_ptr = nullptr;
  if (output_variable_device_tensor_handle != nullptr) {
    output_device_ptr = out_device_handle.device_malloc(output_desc);
  }
  // fprintf(stderr, "===> IN OP Output device allocated=%ld\n",
  // reinterpret_cast<int64_t>(output_device_ptr));

  // fprintf(stderr, "===> IN OP start allocate output pinned ptr.\n");
  void* output_pinned_ptr = nullptr;
  if (output_variable_pinned_tensor_handle != nullptr) {
    output_pinned_ptr = out_pinned_handle.pinned_malloc(output_desc);
  }
  // fprintf(stderr, "===> IN OP Output pinned allocated=%ld\n",
  // reinterpret_cast<int64_t>(output_pinned_ptr));

  // fprintf(stderr, "===> IN OP start allocate output host ptr.\n");
  void* output_host_ptr = nullptr;
  if (output_variable_host_tensor_handle != nullptr) {
    output_host_ptr = out_host_handle.host_malloc(output_desc);
  }
  // fprintf(stderr, "===> IN OP Output host allocated=%ld\n",
  // reinterpret_cast<int64_t>(output_host_ptr));

  WM_CUDA_CHECK_NO_THROW(cudaMemcpyAsync(wholememory_tensor_get_data_pointer(output_fixed_tensor),
                                         temp_buffer_ptr,
                                         output_size,
                                         cudaMemcpyDefault,
                                         cuda_stream));

  if (output_device_ptr)
    WM_CUDA_CHECK_NO_THROW(cudaMemcpyAsync(
      output_device_ptr, temp_buffer_ptr, output_size, cudaMemcpyDefault, cuda_stream));
  if (output_pinned_ptr)
    WM_CUDA_CHECK_NO_THROW(cudaMemcpyAsync(
      output_pinned_ptr, temp_buffer_ptr, output_size, cudaMemcpyDefault, cuda_stream));
  if (output_host_ptr)
    WM_CUDA_CHECK_NO_THROW(cudaMemcpyAsync(
      output_host_ptr, temp_buffer_ptr, output_size, cudaMemcpyDefault, cuda_stream));

  WM_CUDA_DEBUG_SYNC_STREAM(static_cast<cudaStream_t>(stream));

  // fprintf(stderr, "===> IN OP all done.\n");
  return WHOLEMEMORY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
