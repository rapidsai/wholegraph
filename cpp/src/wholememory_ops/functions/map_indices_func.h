/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_tensor.h>

namespace wholememory_ops {

wholememory_error_code_t storage_index2wm_embedding_index(wholememory_tensor_t indices,
                                                          wholememory_tensor_t mapped_indices,
                                                          wholememory_tensor_t allocated_embedding,
                                                          int round_robin_size,
                                                          int64_t stream_int);

}  // namespace wholememory_ops
