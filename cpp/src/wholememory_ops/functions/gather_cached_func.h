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
#include <stdint.h>

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace wholememory_ops {

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
                                            cudaStream_t stream);

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
  cudaStream_t stream);

}  // namespace wholememory_ops
