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
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_tensor.h>

#include "wholememory/embedding_cache.hpp"

namespace wholememory_ops {

/**
 * Direct update cache in local rank, local tensor of wm_raw_memory_embedding is cached by
 * cache_local_data, cache and raw embedding should have same communicator.
 * @param indices : global indices to update, should all in current rank, can have duplicated gids
 * In normal use cases, indices are from alltoallv result
 * @param indice_desc : tensor description of indices, may be gids after alltoallv.
 * @param wm_raw_memory_embedding : the WholeMemory Tensor that is to be cached which stores all
 * embeddings.
 * @param cache_local_data : embedding_cache_local_data of wm_raw_memory_embedding
 * @param cache_set_coverage : cache set coverage
 * @param p_env_fns : env fns
 * @param stream : cudaStream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t update_cache_direct_same_comm(
  void* indices,
  wholememory_array_description_t indice_desc,
  wholememory_tensor_t wm_raw_memory_embedding,
  const wholememory::embedding_cache_local_data* cache_local_data,
  int cache_set_coverage,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);

/**
 * Update cache in local rank, local tensor of wm_raw_memory_embedding is cached by
 * cache_local_data, cache and raw embedding can have same communicator.
 * @param indices : global indices to update, should all in current rank, can have duplicated gids
 * In normal use cases, indices are from alltoallv result
 * @param indice_desc : tensor description of indices, may be gids after alltoallv.
 * @param wm_raw_memory_embedding : the WholeMemory Tensor that is to be cached which stores all
 * embeddings.
 * @param cache_comm : communicator of cache
 * @param embedding_entry_count_per_cache_rank : embedding entries covered by each cache rank
 * @param cache_local_data : embedding_cache_local_data of wm_raw_memory_embedding
 * @param cache_set_coverage : cache set coverage
 * @param p_env_fns : env fns
 * @param stream : cudaStream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t update_cache_different_comm(
  void* indices,
  wholememory_array_description_t indice_desc,
  wholememory_tensor_t wm_raw_memory_embedding,
  wholememory_comm_t cache_comm,
  size_t embedding_entry_count_per_cache_rank,
  const wholememory::embedding_cache_local_data* cache_local_data,
  int cache_set_coverage,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);

wholememory_error_code_t writeback_cache_direct_same_comm(
  wholememory_tensor_t wm_raw_memory_embedding,
  const wholememory::embedding_cache_local_data* cache_local_data,
  int cache_set_coverage,
  bool drop_all,
  cudaStream_t stream);

}  // namespace wholememory_ops
