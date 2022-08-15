/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <nccl.h>

#include <cstddef>
#include <cstdint>

namespace whole_graph {

/*!
 * WholeMemory Initialization, should be called before any other call
 */
void WholeMemoryInit();

/*!
 * Get a prime number no less than value, can be used for hash bucket count selection.
 * @param value: value
 * @return a prime number no less than value, Note, it may be not the first prime number no less than value.
 */
int GetPrimeValue(int value);

/*!
 * WholeMemory Finalization
 */
void WholeMemoryFinalize();

// There are two working modes in WholeMemory.
// One is one process handle multiple GPUs
//   In this mode, there is only one process or different processes works separately.
//   Any function starts with Wmsp are working in this mode.
// The other is one process handle one GPU
//   In this mode, multiple process will works together on a the same set of GPUs.
//   Any function starts with Wmmp are working in this mode.

///////////////////////////////////////////////////////////////////////////////
/*
 * Single Process Functions:
 *    WmspMalloc, WmspMallocHost and WmspFree
 */

/*!
 * Single Process mode malloc
 * @param ptr : return pointer of allocated memory
 * @param size : allocation size
 * @param dev_list : CUDA devices to place memory
 * @param dev_count : CUDA device count
 *        if any of dev_list or dev_count is 0, all CUDA devices will be used
 */
void WmspMalloc(void **ptr, size_t size, const int *dev_list = nullptr, int dev_count = 0);

/*!
 * Single Process mode malloc for host memory
 * @param ptr : return pointer of allocated memory
 * @param size : allocation size
 */
void WmspMallocHost(void **ptr, size_t size);

/*!
 * Single Process mode free
 * @param ptr : memory pointer to free
 */
void WmspFree(void *ptr);

///////////////////////////////////////////////////////////////////////////////
/*
 * Multiple Process Functions
 */

///////////////////////////////////////////////////////////////////////////////

/* Bootstrap */

class BootstrapCommunicatorImpl;

typedef struct {
  ncclUniqueId nccl_unique_id_;
} WmmpUniqueId;

/*
 * Get uniqueId for multi process mode
 */
void WmmpGetUniqueId(WmmpUniqueId *uniqueId);

class BootstrapCommunicator {
 public:
  BootstrapCommunicator();
  ~BootstrapCommunicator();
  /*!
   * Set Rank and Size for Communicator.
   * NOTE this is not world rank and size, but rank and size inside WholeMemory Group.
   *   e.g. process inside one DGX-A100 node may be one WholeMemory Group
   * @param size : communication group size for current process.
   * @param unique_id : WmmpUniqueId created by WmmpGetUniqueId in a single rank.
   * @param rank : group rank of current process
   */
  void InitRank(int size, WmmpUniqueId unique_id, int rank) const;
  /*!
   * Get Rank of current process.
   * @return rank
   */
  int Rank();
  /*!
   * Get Size of collective processes.
   * @return size
   */
  int Size();
  /*!
   * AllToAll with ranks specified
   * @param send_buf : send data buffer
   * @param send_size : send size to each rank
   * @param recv_buf : receive data buffer
   * @param recv_size : receive size from each rank
   */
  void AllToAll(const void *send_buf,
                int send_size,
                void *recv_buf,
                int recv_size) const;
  /*!
   * AllToAllV with ranks specified
   * @param send_bufs : send data buffers
   * @param send_sizes : send sizes to each rank
   * @param recv_bufs : receive data buffers
   * @param recv_sizes : receive sizes from each rank
   */
  void AllToAllV(const void **send_bufs,
                 const int *send_sizes,
                 void **recv_bufs,
                 const int *recv_sizes) const;

  /*!
   * Barrier for all ranks
   */
  void Barrier();

  /*!
   * return underlying ncclComm_t
   * @return : underlying ncclComm_t
   */
  ncclComm_t GetNCCLCommunicator() const;

  /*!
   * return underlying cudaStream_t
   * @return : underlying cudaStream_t
   */
  cudaStream_t GetNCCLStream() const;

  /*!
   * Record into internal cudaEvent_t
   */
  void RecordEvent();

  /*!
   * Synchronize
   */
  void Synchronize();

  /*!
   * return CUDA device of the communicator
   * @return : CUDA device of the communicator
   */
  int CUDADevID() const;

  /*!
   * return whether this is intra-node communicator
   * @return : true if this is intra-node communicator else false
   */
  bool IsIntraNode() const;

  BootstrapCommunicatorImpl *impl_ = nullptr;
};

/*!
 * Create communicator for Multiple Process mode
 * @param size : size of the communicator
 * @param unique_id : unique id get from root rank
 * @param rank : rank of current process
 * @return : BootstrapCommunicator pointer
 */
BootstrapCommunicator *WmmpCreateCommunicator(int size, WmmpUniqueId unique_id, int rank);

/*!
 * Destroy Communicator
 * @param bootstrap_communicator
 */
void WmmpDestroyCommunicator(BootstrapCommunicator *bootstrap_communicator);

/*!
 * WholeMemory Multiple Process mode Finalization
 */
void WmmpFinalize();

/*!
 * WholeMemory Multiple Process mode malloc
 * @param ptr : return pointer of allocated memory
 * @param size : allocation size
 * @param bootstrap_communicator : bootstrap communicator
 */
void WmmpMalloc(void **ptr, size_t size, BootstrapCommunicator *bootstrap_communicator);

/*!
 * WholeMemory Multiple Process mode malloc for host memory
 * @param ptr : return pointer of allocated memory
 * @param size : allocation size
 * @param bootstrap_communicator : bootstrap communicator
 */
void WmmpMallocHost(void **ptr, size_t size, BootstrapCommunicator *bootstrap_communicator);

/*!
 * WholeMemory Multiple Process mode free
 * @param ptr : memory pointer to free
 */
void WmmpFree(void *ptr);

/*!
 * Get bootstrap communicator
 * @param ptr : pointer
 * @return : bootstrap communicator
 */
BootstrapCommunicator *WmmpGetBootstrapCommunicator(const void *ptr);

/*!
 * WmmpGetRankIdxAndRankSizeOfPtr
 * @param rank_idx : return rank_idx
 * @param rank_size : return rank_size
 * @param ptr : input pointer of the WholeMemory
 */
void WmmpGetRankIdxAndRankSizeOfPtr(int *rank_idx, int *rank_size, void *ptr);

/*!
 * WholeMemory Multiple Process mode ptr query accessible device
 * @param ptr : pointer to query
 * @param dev : device to query, -1 means CPU
 * @return true if ptr is accessible from dev, or false if not
 */
bool WmmpCanAccessFrom(void *ptr, int dev);

/*!
 * Helper function, Aggregate local_size and get total_size
 * @param local_size : local size, this value from all ranks will be added together and returned
 * @param offset : return offset of local_size
 * @param bootstrap_communicator : bootstrap communicator
 * @return : the sum of all local_size in ranks
 */
size_t WmmpAggregateSize(size_t local_size, size_t *offset, BootstrapCommunicator *bootstrap_communicator);

/*!
 * Helper function, Barrier
 * @param bootstrap_communicator : bootstrap communicator
 */
void WmmpBarrier(BootstrapCommunicator *bootstrap_communicator);

/*!
 * Helper function, GetRank
 * @param bootstrap_communicator : bootstrap communicator
 */
int64_t WmmpGetRank(BootstrapCommunicator *bootstrap_communicator);

/*!
 * Helper function, GetSize
 * @param bootstrap_communicator : bootstrap communicator
 */
int64_t WmmpGetSize(BootstrapCommunicator *bootstrap_communicator);

/*!
 * AllToAll with ranks specified
 * @param send_buf : send data buffer
 * @param send_size : send size to each rank
 * @param recv_buf : receive data buffer
 * @param recv_size : receive size from each rank
 * @param bootstrap_communicator : bootstrap communicator

 */
void WmmpAllToAll(const void *send_buf,
                  int send_size,
                  void *recv_buf,
                  int recv_size,
                  BootstrapCommunicator *bootstrap_communicator);

/*!
 * Helper function, given job size total_size, ranks and rank_count, distrubute total_size to all ranks
 *   and returns offset for current rank and local_size.
 * @param total_size : total job size
 * @param local_size : output, pointer of local job size to process
 * @param bootstrap_communicator : bootstrap communicator
 * @return : offset for local rank
 */
size_t WmmpCollOffsetAndSize(size_t total_size, size_t *local_size, BootstrapCommunicator *bootstrap_communicator);

/*!
 * Helper function, get the rank and size of a WholeMemory
 * @param ptr : pointer or WholeChunkedMemory_t, should be base address.
 * @param rank : [Output] rank in the WholeMemory
 * @param size : [Output] size of the WholeMemory
 */
void GetRankAndSize(void *ptr, int *rank, int *size);

class WmmpAWBarrier;

typedef WmmpAWBarrier *WmmpAWBarrier_t;

WmmpAWBarrier_t WmmpCreateAWBarrier(BootstrapCommunicator *bootstrap_communicator);

void WmmpDestroyAWBarrier(WmmpAWBarrier_t barrier);

void WmmpAWBarrierArrive(WmmpAWBarrier_t barrier, cudaStream_t stream);

void WmmpAWBarrierWait(WmmpAWBarrier_t barrier, cudaStream_t stream);

void WmmpAWBarrierArriveAndWait(WmmpAWBarrier_t barrier, cudaStream_t stream);

}// namespace whole_graph