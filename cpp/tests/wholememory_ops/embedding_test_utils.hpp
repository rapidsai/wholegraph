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

#include <wholememory/env_func_ptrs.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace wholememory_ops {
namespace testing {

void device_matrix_type_cast(void* dst,
                             wholememory_matrix_description_t dst_desc,
                             const void* src,
                             wholememory_matrix_description_t src_desc,
                             cudaStream_t stream);

void device_array_type_cast(void* dst,
                            wholememory_array_description_t dst_desc,
                            const void* src,
                            wholememory_array_description_t src_desc,
                            cudaStream_t stream);

void device_random_init_local_embedding_table(wholememory_handle_t embedding_handle,
                                              wholememory_matrix_description_t embedding_desc,
                                              cudaStream_t stream);

void device_get_expected_embedding(void* output,
                                   wholememory_matrix_description_t output_desc,
                                   wholememory_dtype_t embedding_dtype,
                                   void* indices,
                                   wholememory_array_description_t indices_desc,
                                   wholememory_env_func_t* p_env_fns,
                                   cudaStream_t stream);

/**
 * random generate indices from [0, max_indices)
 * @param indices : pointer of output
 * @param indices_desc : description of output
 * @param max_indices : max_indices
 */
void host_random_init_indices(void* indices,
                              wholememory_array_description_t indices_desc,
                              int64_t max_indices);

void host_check_embedding_same(void* host_embedding,
                               wholememory_matrix_description_t embedding_desc,
                               void* host_reference,
                               wholememory_matrix_description_t reference_desc);

void host_random_init_float(float* data, int64_t len, float max_value, float min_value);

}  // namespace testing
}  // namespace wholememory_ops
