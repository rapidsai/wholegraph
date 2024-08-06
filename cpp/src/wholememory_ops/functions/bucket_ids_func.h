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

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace wholememory_ops {

wholememory_error_code_t bucket_ids_for_ranks(void* indices,
                                              wholememory_array_description_t indice_desc,
                                              int64_t* dev_rank_id_count_ptr,
                                              size_t* embedding_entry_offsets,
                                              int world_size,
                                              cudaDeviceProp* prop,
                                              cudaStream_t stream);

}  // namespace wholememory_ops
