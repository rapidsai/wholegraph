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

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to WholeMemory Embedding Cache Policy
 *
 * An Opaque handle to WholeMemory Embedding Cache Policy
 */
typedef struct wholememory_embedding_cache_policy_* wholememory_embedding_cache_policy_t;

/**
 * @brief Opaque handle to WholeMemory Embedding Optimizer
 *
 * An Opaque handle to WholeMemory Embedding Optimizer
 */
typedef struct wholememory_embedding_optimizer_* wholememory_embedding_optimizer_t;

/**
 * @brief Opaque handle to WholeMemory Embedding
 *
 * An Opaque handle to WholeMemory Embedding
 */
typedef struct wholememory_embedding_* wholememory_embedding_t;

/**
 * @enum wholememory_access_type_t
 * @brief defines access type of WholeMemory Embedding
 */
enum wholememory_access_type_t {
  WHOLEMEMORY_AT_NONE = 0,  /*!< Not defined */
  WHOLEMEMORY_AT_READONLY,  /*!< Only have readonly access to the WholeMemory */
  WHOLEMEMORY_AT_READWRITE, /*!< May have write access to the WholeMemory */
};

/**
 * @enum wholememory_optimizer_type_t
 * @brief defines optimizer type for WholeMemory Embedding
 */
enum wholememory_optimizer_type_t {
  WHOLEMEMORY_OPT_NONE = 0,  /*!< No optimizer needed */
  WHOLEMEMORY_OPT_SGD,       /*!< Use SGD optimizer */
  WHOLEMEMORY_OPT_LAZY_ADAM, /*!< Use Lazy Adam optimizer */
  WHOLEMEMORY_OPT_RMSPROP,   /*!< Use RMSProp optimizer */
  WHOLEMEMORY_OPT_ADAGRAD,   /*!< Use AdaGrad optimizer */
};

/**
 * Create Optimizer
 * @param optimizer : Returned wholememory_embedding_optimizer_t
 * @param optimizer_type : Optimizer type
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_create_embedding_optimizer(
  wholememory_embedding_optimizer_t* optimizer, wholememory_optimizer_type_t optimizer_type);

/**
 * Set parameter for optimizer.
 * @param optimizer : Optimizer to set parameter
 * @param parameter_name : parameter name
 * @param value : parameter value
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_optimizer_set_parameter(
  wholememory_embedding_optimizer_t optimizer, const char* parameter_name, void* value);

/**
 * Destroy optimizer
 * @param optimizer : optimizer to destroy.
 */
void wholememory_destroy_embedding_optimizer(wholememory_embedding_optimizer_t optimizer);

/**
 * Create WholeMemory Embedding Cache Policy
 * @param cache_policy : Returned wholememory_embedding_cache_policy_t
 * @param cache_level_comm : At which level to cache the full embedding. In most cases it should be
 * same as wholememory_embedding_t's comm. If access_type is WHOLEMEMORY_AT_READONLY, it can be
 * different for multiple readonly caches. E.g. for a multi-node WHOLEMEMORY_MT_DISTRIBUTED
 * WHOLEMEMORY_AT_READONLY embedding, it can have a intra-node WHOLEMEMORY_MT_CHUNKED cache. or a
 * multi-node WHOLEMEMORY_MT_DISTRIBUTED cache.
 * @param memory_type : Memory Type of the underlying WholeMemory for cache
 * @param memory_location : Memory Location of the underlying WholeMemory for cache
 * @param access_type : ReadOnly or ReadWrite
 * @param cache_ratio : suggested cache ratio, values should be in range [1.0 / 512, 1.0]
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_create_embedding_cache_policy(
  wholememory_embedding_cache_policy_t* cache_policy,
  wholememory_comm_t cache_level_comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location,
  wholememory_access_type_t access_type,
  float cache_ratio);

/**
 * Destroy WholeMemory Embedding Cache Policy
 * @param cache_policy : WholeMemory Embedding Cache Policy to destroy.
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_destroy_embedding_cache_policy(
  wholememory_embedding_cache_policy_t cache_policy);

/**
 * Create WholeMemory Embedding
 * @param wholememory_embedding : Returned wholememory_embedding_t
 * @param embedding_tensor_description : Description of the embedding, sizes and dtype used, stride
 * and storage_offset ignored. Must be matrix
 * @param comm : WholeMemory Communicator
 * @param memory_type : Memory Type of the underlying WholeMemory
 * @param memory_location : Memory Location of the underlying WholeMemory
 * @param cache_policy : Cache policy for this embedding, if don't use cache, use nullptr
 * @param embedding_entry_partition: Embedding entry count of each rank, the length must be
 * world_size
 * @param user_defined_sms : User-defined sms number for raw embedding gather/scatter
 * @param round_robin_size : continuous embedding size in each rank under round-robin shard mode
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_create_embedding(
  wholememory_embedding_t* wholememory_embedding,
  wholememory_tensor_description_t* embedding_tensor_description,
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location,
  wholememory_embedding_cache_policy_t cache_policy,
  size_t* embedding_entry_partition = nullptr,
  int user_defined_sms              = -1,
  int round_robin_size              = 0);

/**
 * Destroy WholeMemory Embedding
 * @param wholememory_embedding : WholeMemory Embedding to destroy
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_destroy_embedding(
  wholememory_embedding_t wholememory_embedding);

/**
 * Get WholeMemory Tensor from WholeMemory Embedding.
 * @param wholememory_embedding : WholeMemory Embedding
 * @return : WholeMemory Tensor
 */
wholememory_tensor_t wholememory_embedding_get_embedding_tensor(
  wholememory_embedding_t wholememory_embedding);

/**
 * Set Optimizer for WholeMemory Embedding
 * @param wholememory_embedding : WholeMemory Embedding
 * @param optimizer : Optimizer to be set
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_embedding_set_optimizer(
  wholememory_embedding_t wholememory_embedding, wholememory_embedding_optimizer_t optimizer);

/**
 * Gather from WholeMemory Embedding
 * @param wholememory_embedding : WholeMemory Embedding
 * @param indices : indices to gather
 * @param output : output tensor
 * @param adjust_cache : if we should adjust cache in this gather
 * @param p_env_fns : env fns
 * @param stream_int : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_embedding_gather(wholememory_embedding_t wholememory_embedding,
                                                      wholememory_tensor_t indices,
                                                      wholememory_tensor_t output,
                                                      bool adjust_cache,
                                                      wholememory_env_func_t* p_env_fns,
                                                      int64_t stream_int);

/**
 * Gather backward for WholeMemory Embedding
 * @param wholememory_embedding : WholeMemory Embedding
 * @param indices : indices to gather
 * @param grads : gradient of output tensor
 * @param adjust_cache : if we should adjust cache in this gather
 * @param lr : learning rate of current step.
 * @param p_env_fns : env fns
 * @param stream_int : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_embedding_gather_gradient_apply(
  wholememory_embedding_t wholememory_embedding,
  wholememory_tensor_t indices,
  wholememory_tensor_t grads,
  bool adjust_cache,
  float lr,
  wholememory_env_func_t* p_env_fns,
  int64_t stream_int);

/**
 * Get optimizer internal state names
 * @param wholememory_embedding : WholeMemory Embedding
 * @return : nullptr terminated names.
 */
const char* const* wholememory_embedding_get_optimizer_state_names(
  wholememory_embedding_t wholememory_embedding);

/**
 * Get optimizer internal state
 * @param wholememory_embedding : WholeMemory Embedding
 * @param name : state name
 * @return : internal state, nullptr for not exist.
 */
wholememory_tensor_t wholememory_embedding_get_optimizer_state(
  wholememory_embedding_t wholememory_embedding, const char* name);

/**
 * Writeback all cache WholeMemory Embedding
 * @param wholememory_embedding : WholeMemory Embedding
 * @param stream_int : CUDA stream to use.
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_embedding_writeback_cache(
  wholememory_embedding_t wholememory_embedding, int64_t stream_int);

/**
 * Drop all cache in WholeMemory Embedding
 * @param wholememory_embedding : WholeMemory Embedding
 * @param stream_int : CUDA stream to use.
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_embedding_drop_all_cache(
  wholememory_embedding_t wholememory_embedding, int64_t stream_int);

#ifdef __cplusplus
}
#endif
