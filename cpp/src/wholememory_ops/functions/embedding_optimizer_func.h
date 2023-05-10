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

#include <cuda_runtime_api.h>

#include <wholememory/wholememory.h>
#include <wholememory/wholememory_tensor.h>

namespace wholememory_ops {

void set_memory_to_float_value(float* data_ptr, float value, size_t elt_count, cudaStream_t stream);

wholememory_error_code_t sgd_optimizer_step(wholememory_tensor_t indices,
                                            wholememory_tensor_t grads,
                                            wholememory_tensor_t local_embedding,
                                            wholememory_tensor_t local_embedding_cache_tag,
                                            wholememory_tensor_t local_embedding_cache_data,
                                            int64_t local_entry_offset,
                                            int cache_set_coverage,
                                            float weight_decay,
                                            float lr,
                                            cudaStream_t stream);

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
                                                  cudaStream_t stream);

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
                                                 cudaStream_t stream);

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
                                                 cudaStream_t stream);

}  // namespace wholememory_ops
