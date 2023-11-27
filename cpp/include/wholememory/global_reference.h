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

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Global reference of a WholeMemory object
 *
 * A global reference is for Continuous of Chunked WholeMemory Type, in these types, each rank can
 * directly access all memory from all ranks. The global reference is used to do this direct access.
 */
struct wholememory_gref_t {
  void* pointer; /*!< pointer to data for CONTINUOUS WholeMemory or pointer to data pointer array
                    for CHUNKED WholeMemory */
  size_t
    stride; /*!< must be 0 for CONTINUOUS WholeMemory or memory size in byte for each pointer */
};

/**
 * @brief Create global reference for continuous memory
 * @param ptr : pointer to the memory
 * @return : wholememory_gref_t
 */
wholememory_gref_t wholememory_create_continuous_global_reference(void* ptr);

struct wholememory_nvshmem_ref_t {
  void* pointer;
  size_t stride;
  int world_rank;
  int world_size;
};

#ifdef __cplusplus
}
#endif
