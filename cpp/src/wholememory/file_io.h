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

#include <wholememory/wholememory.h>

namespace wholememory {

wholememory_error_code_t load_file_to_handle(wholememory_handle_t wholememory_handle,
                                             size_t memory_offset,
                                             size_t memory_entry_stride,
                                             size_t entry_size,
                                             const char** file_names,
                                             int file_count,
                                             int round_robin_size) noexcept;

wholememory_error_code_t store_handle_to_file(wholememory_handle_t wholememory_handle,
                                              size_t memory_offset,
                                              size_t memory_entry_stride,
                                              size_t entry_size,
                                              const char* local_file_name) noexcept;

}  // namespace wholememory
