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

#include <stdio.h>
#include <unistd.h>

#include <wholememory/global_reference.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief WholeMemory Error Code definition
 *
 * Defines error code of WholeMemory library.
 */
enum wholememory_error_code_t {
  WHOLEMEMORY_SUCCESS = 0,         /*!< success */
  WHOLEMEMORY_UNKNOW_ERROR,        /*!< unknown error */
  WHOLEMEMORY_NOT_IMPLEMENTED,     /*!< method is not implemented */
  WHOLEMEMORY_LOGIC_ERROR,         /*!< logic error */
  WHOLEMEMORY_CUDA_ERROR,          /*!< CUDA error */
  WHOLEMEMORY_COMMUNICATION_ERROR, /*!< communication error */
  WHOLEMEMORY_INVALID_INPUT,       /*!< input is invalid, e.g. nullptr */
  WHOLEMEMORY_INVALID_VALUE,       /*!< input value is invalid */
  WHOLEMEMORY_OUT_OF_MEMORY,       /*!< out of memory */
  WHOLEMEMORY_NOT_SUPPORTED,       /*!< not supported */
  WHOLEMEMORY_SYSTEM_ERROR,        /*!< system error>*/
};

#define WHOLEMEMORY_RETURN_ON_FAIL(X)                                                 \
  do {                                                                                \
    auto err = X;                                                                     \
    if (err != WHOLEMEMORY_SUCCESS) {                                                 \
      const char* error_str = #X;                                                     \
      fprintf(stderr, "File %s line %d %s failed.\n", __FILE__, __LINE__, error_str); \
      return err;                                                                     \
    }                                                                                 \
  } while (0)

/**
 * @brief Memory Type of WholeMemory
 *
 * Memory Type is the Memory Address Mapping Type of WholeMemory
 */
enum wholememory_memory_type_t {
  WHOLEMEMORY_MT_NONE = 0,    /*!< Not defined.  */
  WHOLEMEMORY_MT_CONTINUOUS,  /*!< Memory from all ranks are mapped in continuous address space */
  WHOLEMEMORY_MT_CHUNKED,     /*!< Memory from all ranks are mapped in chunked address space */
  WHOLEMEMORY_MT_DISTRIBUTED, /*!< Memory from other ranks are not mapped. */
  WHOLEMEMORY_MT_HIERARCHY,   /*!< Memory from other ranks are mapped in hierarchy address space */
};

/**
 * @brief Memory Location of WholeMemory
 *
 * Memory Location of WholeMemory can be host or device.
 */
enum wholememory_memory_location_t {
  WHOLEMEMORY_ML_NONE = 0, /*!< Not defined */
  WHOLEMEMORY_ML_DEVICE,   /*!< Device Memory */
  WHOLEMEMORY_ML_HOST,     /*!< Host Memory */
};

enum wholememory_distributed_backend_t {
  WHOLEMEMORY_DB_NONE = 0, /*!< Not defined */
  WHOLEMEMORY_DB_NCCL,
  WHOLEMEMORY_DB_NVSHMEM,
};

enum LogLevel {
  LEVEL_FATAL = 0, /*!< Fatal */
  LEVEL_ERROR,     /*!< Error */
  LEVEL_WARN,      /*!< Warn */
  LEVEL_INFO,      /*!< Info */
  LEVEL_DEBUG,     /*!< Debug*/
  LEVEL_TRACE      /*!< Trace */
};

#define WHOLEMEMORY_SPILT_NO_COLOR -1
/**
 * Initialize WholeMemory library
 * @param flags : reserved should be 0
 * @param log_level : wholememory log level, the default level is "info"
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_init(unsigned int flags, LogLevel log_level = LEVEL_INFO);

/**
 * Finalize WholeMemory library
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_finalize();

/**
 * @brief Opaque handle to communicator
 *
 * An Opaque handle to communicator
 */
typedef struct wholememory_comm_* wholememory_comm_t;

struct clique_info_t {
  int is_in_clique;  // is_in_clique >0 means the gpu belongs to  a mnnvl domain
  int clique_first_rank;
  int clique_rank;      // the rank of gpu in a mnnvl domain
  int clique_rank_num;  // the num of gpu in the mnnvl domain
  int clique_id;        // the id of clique
  int clique_num;       // the num of clique in the comm domain.
};

#define WHOLEMEMORY_UNIQUE_ID_BYTES (128)
/**
 * @brief Unique ID for WholeMemory Communicators
 *
 * An Opaque handle to WholeMemory Communicators, exposes as char array.
 * Underlying implementation may be ncclUniqueId_t
 */
struct wholememory_unique_id_t {
  char internal[WHOLEMEMORY_UNIQUE_ID_BYTES];
};

/**
 * Create UniqueID for WholeMemory Communicator
 * @param unique_id : returned UniqueID
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_create_unique_id(wholememory_unique_id_t* unique_id);

/**
 * Create WholeMemory Communicator
 * @param comm : returned WholeMemory Communicator
 * @param unique_id : UniqueID
 * @param rank : rank of this process.
 * @param size : number of processes in this Communicator
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_create_communicator(wholememory_comm_t* comm,
                                                         wholememory_unique_id_t unique_id,
                                                         int rank,
                                                         int size);

/**
 * Split WholeMemory Communicator
 * @param new_comm: returned the splited wholeMemory Communicator
 * @param comm: WholeMemory Communicator to split
 * @param color: color value to split communicator,Ranks which pass the same color value will be
 * part of the same group; color must be a non-negative value. If it is passed as
 * WHOLEMEMORY_SPLIT_NOCOLOR, it means that the rank will not be part of any group, therefore
 * returning NULL as newcomm.
 * @param key: key value to split communicator,the value of key will determine the
 * rank order, and the smaller key means the smaller rank in new communicator. If keys are equal
 * between ranks, then the rank in the original communicator will be used to order ranks.
 * @return : wholememory_error_code_t

*/
wholememory_error_code_t wholememory_split_communicator(wholememory_comm_t* new_comm,
                                                        wholememory_comm_t comm,
                                                        int color,
                                                        int key);
/**
 * Destroy WholeMemory Communicator
 * @param comm : WholeMemory Communicator to destroy
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_destroy_communicator(wholememory_comm_t comm);

/**
 * Check if combination of WholeMemory type and location is supported in the communicator
 * @param comm : WholeMemory Communicator
 * @param memory_type : WholeMemory type
 * @param memory_location : WholeMemory Location
 * @return WHOLEMEMORY_SUCCESS if supported else WHOLEMEMORY_NOT_SUPPORTED
 */
wholememory_error_code_t wholememory_communicator_support_type_location(
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location);

/**
 * Get the rank of current process in the WholeMemory Communicator
 * @param rank : returned rank
 * @param comm : WholeMemory Communicator
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_communicator_get_rank(int* rank, wholememory_comm_t comm);

/**
 * Get the size of WholeMemory Communicator
 * @param size : returned size
 * @param comm : WholeMemory Communicator
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_communicator_get_size(int* size, wholememory_comm_t comm);

/**
 * Get the local rank size of current process in the WholeMemory Communicator
 * @param local_size : returned local rank size
 * @param comm : WholeMemory Communicator
 * @return : wholememory_error_code_t
 */

wholememory_error_code_t wholememory_communicator_get_local_size(int* local_size,
                                                                 wholememory_comm_t comm);

/**
 * Get the clique info of WholeMemory Communicator
 * @param clique_info : returned clique info
 * @param comm : WholeMemory Communicator
 * @return : wholememory_error_code_t
 */

wholememory_error_code_t wholememory_communicator_get_clique_info(clique_info_t* clique_info,
                                                                  wholememory_comm_t comm);

bool wholememory_communicator_is_bind_to_nvshmem(wholememory_comm_t comm);

wholememory_error_code_t wholememory_communicator_set_distributed_backend(
  wholememory_comm_t comm, wholememory_distributed_backend_t distributed_backend);

wholememory_distributed_backend_t wholememory_communicator_get_distributed_backend(
  wholememory_comm_t comm);
/**
 * Barrier on WholeMemory Communicator
 * @param comm : WholeMemory Communicator
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_communicator_barrier(wholememory_comm_t comm);

/**
 * @brief Opaque handle to WholeMemory
 *
 * An Opaque handle to WholeMemory
 */
typedef struct wholememory_handle_* wholememory_handle_t;

/**
 * Malloc WholeMemory
 * @param wholememory_handle_ptr : returned WholeMemory Handle
 * @param total_size : total allocated size in bytes.
 * @param comm : WholeMemory Communicator
 * @param memory_type : WholeMemory type
 * @param memory_location : memory location, host or device
 * @param data_granularity : granularity size of data, which is guaranteed not to be partitioned.
 * @param rank_entry_partition : entry count of each rank (size of entry equal to data_granularity)
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_malloc(wholememory_handle_t* wholememory_handle_ptr,
                                            size_t total_size,
                                            wholememory_comm_t comm,
                                            wholememory_memory_type_t memory_type,
                                            wholememory_memory_location_t memory_location,
                                            size_t data_granularity,
                                            size_t* rank_entry_partition = nullptr);

/**
 * Free allocated WholeMemory Handle
 * @param wholememory_handle : WholeMemory Handle to free
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_free(wholememory_handle_t wholememory_handle);

/**
 * Get underlying WholeMemory Communicator from WholeMemory Handle
 * @param comm : returned WholeMemory Communicator
 * @param wholememory_handle : WholeMemory Handle
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_get_communicator(wholememory_comm_t* comm,
                                                      wholememory_handle_t wholememory_handle);

/**
 * Get underlying Wholememory Local Communicator for "Hierarchy" memory type from WholeMemory Handle
 * @param comm : returned Local WholeMemory Communicator
 * @param wholememory_handle : WholeMemory Handle
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_get_local_communicator(
  wholememory_comm_t* comm, wholememory_handle_t wholememory_handle);

/**
 * Get underlying Wholememory Cross Communicator for "Hierarchy" memory type from WholeMemory Handle
 * One comminicator includes all rank with a same local id from different nodes
 * @param comm : returned Cross WholeMemory Communicator
 * @param wholememory_handle : WholeMemory Handle
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_get_cross_communicator(
  wholememory_comm_t* comm, wholememory_handle_t wholememory_handle);

/**
 * Get WholeMemory Type
 * @param wholememory_handle : WholeMemory Handle
 * @return : WholeMemory Type
 */
wholememory_memory_type_t wholememory_get_memory_type(wholememory_handle_t wholememory_handle);

/**
 * Get WholeMemory Location
 * @param wholememory_handle : WholeMemory Handle
 * @return : WholeMemory Location
 */
wholememory_memory_location_t wholememory_get_memory_location(
  wholememory_handle_t wholememory_handle);

wholememory_distributed_backend_t wholememory_get_distributed_backend(
  wholememory_handle_t wholememory_handle);

/**
 * Get total size of WholeMemory
 * @param wholememory_handle : WholeMemory Handle
 * @return : total size
 */
size_t wholememory_get_total_size(wholememory_handle_t wholememory_handle);

/**
 * Get data granularity of WholeMemory Handle
 * @param wholememory_handle : WholeMemory Handle
 * @return : data granularity size
 */
size_t wholememory_get_data_granularity(wholememory_handle_t wholememory_handle);

/**
 * Get local memory from WholeMemory Handle of current rank, local memory has direct access to the
 * memory. But local memory doesn't have to be on local GPU.
 * @param local_ptr : returned local memory pointer
 * @param local_size : returned local memory size
 * @param local_offset : returned local memory offset from WholeMemory
 * @param wholememory_handle : WholeMemory Handle
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_get_local_memory(void** local_ptr,
                                                      size_t* local_size,
                                                      size_t* local_offset,
                                                      wholememory_handle_t wholememory_handle);

/**
 * Get local memory size from WholeMemory Handle of current rank
 * @param local_size : returned local memory size
 * @param wholememory_handle : WholeMemory Handle
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_get_local_size(size_t* local_size,
                                                    wholememory_handle_t wholememory_handle);

/**
 * Get local memory offset from WholeMemory Handle of current rank
 * @param local_offset : returned local memory offset
 * @param wholememory_handle : WholeMemory Handle
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_get_local_offset(size_t* local_offset,
                                                      wholememory_handle_t wholememory_handle);

/**
 * Get local memory of specified rank from WholeMemory Handle
 * @param rank_memory_ptr : returned local memory pointer of specified rank
 * @param rank_memory_size : returned local memory size of specified rank
 * @param rank_memory_offset : returned local memory offset of specified rank from WholeMemory
 * @param rank : rank specified
 * @param wholememory_handle : WholeMemory Handle
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_get_rank_memory(void** rank_memory_ptr,
                                                     size_t* rank_memory_size,
                                                     size_t* rank_memory_offset,
                                                     int rank,
                                                     wholememory_handle_t wholememory_handle);

/**
 * Get the equal partition plan WholeMemory uses by default
 * @param entry_per_rank : returned entry count per rank
 * @param total_entry_count : total entry count
 * @param world_size : communicator world size
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_equal_entry_partition_plan(size_t* entry_per_rank,
                                                                size_t total_entry_count,
                                                                int world_size);

/**
 * Get global memory pointer from WholeMemory Handle.
 * Only Continuous memory type or Chunked Host memory has global pointer.
 * @param global_ptr : returned pointer of WholeMemory
 * @param wholememory_handle : WholeMemory Handle
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_get_global_pointer(void** global_ptr,
                                                        wholememory_handle_t wholememory_handle);

/**
 * Get global reference from WholeMemory Handle
 * WholeMemory global reference is common data structure for Continuous and Chunked Memory Types.
 * @param wholememory_gref : returned WholeMemory global reference
 * @param wholememory_handle : WholeMemory Handle
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_get_global_reference(wholememory_gref_t* wholememory_gref,
                                                          wholememory_handle_t wholememory_handle);

/**
 * Get memory size of each rank from WholeMemory Handle
 * @param rank_mem_sizes : returned memory size of each rank
 * @param wholememory_handle : WholeMemory Handle
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_get_rank_partition_sizes(
  size_t* rank_mem_sizes, wholememory_handle_t wholememory_handle);

/**
 * Get memory offset of each rank from WholeMemory Handle
 * @param rank_mem_offsets : returned memory offset of each rank
 * @param wholememory_handle : WholeMemory Handle
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_get_rank_partition_offsets(
  size_t* rank_mem_offsets, wholememory_handle_t wholememory_handle);

/**
 * Fork a new process and get device count. Should be called before other CUDA call
 * @return : CUDA device count, -1 on error
 */
int fork_get_device_count();

/**
 * Load WholeMemory from binary files, all rank should be called together
 * @param wholememory_handle : WholeMemory Handle
 * @param memory_offset : load to memory offset
 * @param memory_entry_size : entry size of WholeMemory
 * @param file_entry_size : entry size in file, should be less than or equal to memory_entry_size
 * @param file_names : file names, all binary files will be logically concatenated and loaded.
 * @param file_count : number of files.
 * @param round_robin_size : continuous embedding number for a rank under round-robin shard mode
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_load_from_file(wholememory_handle_t wholememory_handle,
                                                    size_t memory_offset,
                                                    size_t memory_entry_size,
                                                    size_t file_entry_size,
                                                    const char** file_names,
                                                    int file_count,
                                                    int round_robin_size);

/**
 * Store local WholeMemory to file, this should be called by all ranks, with different
 * local_file_name.
 * @param wholememory_handle : WholeMemory Handle
 * @param memory_offset : memory offset to store
 * @param memory_entry_stride : entry size of WholeMemory
 * @param file_entry_size : entry size in file, should be less than or equal to memory_entry_size
 * @param local_file_name : local file to store to
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_store_to_file(wholememory_handle_t wholememory_handle,
                                                   size_t memory_offset,
                                                   size_t memory_entry_stride,
                                                   size_t file_entry_size,
                                                   const char* local_file_name);

/**
 * @param comm : WholeMemory Comm
 * @return : bool
 */
bool wholememory_is_intranode_communicator(wholememory_comm_t comm);

bool wholememory_is_intra_mnnvl_communicator(wholememory_comm_t comm);
bool wholememory_is_build_with_nvshmem();
#ifdef WITH_NVSHMEM_SUPPORT
wholememory_error_code_t wholememory_get_nvshmem_reference(
  wholememory_nvshmem_ref_t* wholememory_nvshmem_ref, wholememory_handle_t wholememory_handle);
#endif

#ifdef __cplusplus
}
#endif
