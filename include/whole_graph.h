#pragma once

#include <cstddef>

namespace whole_graph {

/*!
 * WholeMemory Initialization, should be called before any other call
 */
void WholeMemoryInit();

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
void WmspMalloc(void** ptr, size_t size, const int* dev_list = nullptr, int dev_count = 0);

/*!
 * Single Process mode malloc for host memory
 * @param ptr : return pointer of allocated memory
 * @param size : allocation size
 */
void WmspMallocHost(void** ptr, size_t size);

/*!
 * Single Process mode free
 * @param ptr : memory pointer to free
 */
void WmspFree(void* ptr);

///////////////////////////////////////////////////////////////////////////////
/*
 * Multiple Process Functions
 */

class CollectiveCommunicator {
 public:
  virtual ~CollectiveCommunicator() = default;
  /*!
   * Set Rank and Size for Communicator.
   * NOTE this is not world rank and size, but rank and size inside WholeMemory Group.
   *   e.g. process inside one DGX-A100 node may be one WholeMemory Group
   * @param rank : group rank of current process
   * @param size : communication group size for current process.
   */
  virtual void SetRankAndSize(int rank, int size) = 0;
  /*!
   * Get Rank of current process.
   * @return rank
   */
  virtual int Rank() = 0;
  /*!
   * Get Size of collective processes.
   * @return size
   */
  virtual int Size() = 0;
  /*!
   * AllToAll
   * @param send_buf : send data buffer
   * @param send_size : send size to each rank
   * @param recv_buf : receive data buffer
   * @param recv_size : receive size from each rank
   */
  void AllToAll(const void* send_buf, int send_size, void* recv_buf, int recv_size) {
    AllToAll(send_buf, send_size, recv_buf, recv_size, nullptr, 0);
  }
  /*!
   * AllToAll with ranks specified
   * @param send_buf : send data buffer
   * @param send_size : send size to each rank
   * @param recv_buf : receive data buffer
   * @param recv_size : receive size from each rank
   * @param ranks : ranks participate in this AllToAll, nullptr means all ranks
   * @param rank_count : rank count participate in this AllToAll, 0 means all ranks
   */
  virtual void AllToAll(const void* send_buf, int send_size, void* recv_buf, int recv_size, const int* ranks, int rank_count) = 0;
  /*!
   * AllToAllV
   * @param send_bufs : send data buffers
   * @param send_sizes : send sizes to each rank
   * @param recv_bufs : receive data buffers
   * @param recv_sizes : receive sizes from each rank
   */
  void AllToAllV(const void** send_bufs, const int* send_sizes, void** recv_bufs, const int* recv_sizes) {
    return AllToAllV(send_bufs, send_sizes, recv_bufs, recv_sizes, nullptr, 0);
  }
  /*!
   * AllToAllV with ranks specified
   * @param send_bufs : send data buffers
   * @param send_sizes : send sizes to each rank
   * @param recv_bufs : receive data buffers
   * @param recv_sizes : receive sizes from each rank
   * @param ranks : ranks participate in this AllToAll, nullptr means all ranks
   * @param rank_count : rank count participate in this AllToAll, 0 means all ranks
   */
  virtual void AllToAllV(const void** send_bufs, const int* send_sizes, void** recv_bufs, const int* recv_sizes, const int* ranks, int rank_count) = 0;
  /*!
   * Barrier for all ranks
   */
  void Barrier() {
    Barrier(nullptr, 0);
  }
  /*!
   * Barrier with ranks specified, Default Barrier implementation calls AllToAll with small data
   * @param ranks : ranks participate in this Barrier, nullptr means all ranks
   * @param rank_count : rank count participate in this Barrier, 0 means all ranks
   */
  virtual void Barrier(const int* ranks, int rank_count);
 protected:
  CollectiveCommunicator() = default;
};

/*!
 * WholeMemory Multiple Process mode Initialization, should be called before any other Wmmp* call
 * @param group_rank : group rank of current process
 * @param group_size : communication group size for current process
 * @param collective_communicator : collective communicator to use, if nullptr, will use MPI communicator
 */
void WmmpInit(int group_rank, int group_size, CollectiveCommunicator* collective_communicator = nullptr);

/*!
 * WholeMemory Multiple Process mode Finalization
 */
void WmmpFinalize();

/*!
 * WholeMemory Multiple Process mode malloc
 * @param ptr : return pointer of allocated memory
 * @param size : allocation size
 * @param ranks : ranks participate in this allocation, nullptr means all ranks
 * @param rank_count : rank count participate in this allocation, 0 means all ranks
 */
void WmmpMalloc(void** ptr, size_t size, const int* ranks = nullptr, int rank_count = 0);

/*!
 * WholeMemory Multiple Process mode malloc for host memory
 * @param ptr : return pointer of allocated memory
 * @param size : allocation size
 * @param ranks : ranks participate in this allocation, nullptr means all ranks
 * @param rank_count : rank count participate in this allocation, 0 means all ranks
 */
void WmmpMallocHost(void** ptr, size_t size, const int* ranks = nullptr, int rank_count = 0);

/*!
 * WholeMemory Multiple Process mode free
 * @param ptr : memory pointer to free
 */
void WmmpFree(void* ptr);

/*!
 * WholeMemory Multiple Process mode ptr query accessible device
 * @param ptr : pointer to query
 * @param dev : device to query, -1 means CPU
 * @return true if ptr is accessible from dev, or false if not
 */
bool WmmpCanAccessFrom(void* ptr, int dev);

/*!
 * Helper function, Aggregate local_size and get total_size
 * @param local_size : local size, this value from all ranks will be added together and returned
 * @param offset : return offset of local_size
 * @param ranks : ranks participate in this aggregation, nullptr means all ranks
 * @param rank_count : rank count participate in this aggregation, 0 means all ranks
 * @return : the sum of all local_size in ranks
 */
size_t WmmpAggregateSize(size_t local_size, size_t* offset, const int* ranks = nullptr, int rank_count = 0);

/*!
 * Helper function, Barrier
 * @param ranks : ranks participate in this barrier, nullptr means all ranks
 * @param rank_count : rank count participate in this barrier, 0 means all ranks
 */
void WmmpBarrier(const int* ranks = nullptr, int rank_count = 0);

/*!
 * AllToAll with ranks specified
 * @param send_buf : send data buffer
 * @param send_size : send size to each rank
 * @param recv_buf : receive data buffer
 * @param recv_size : receive size from each rank
 * @param ranks : ranks participate in this AllToAll, nullptr means all ranks
 * @param rank_count : rank count participate in this AllToAll, 0 means all ranks
 */
void WmmpAllToAll(const void* send_buf, int send_size, void* recv_buf, int recv_size, const int* ranks = nullptr, int rank_count = 0);

/*!
 * Helper function, given job size total_size, ranks and rank_count, distrubute total_size to all ranks
 *   and returns offset for current rank and local_size.
 * @param total_size : total job size
 * @param local_size : output, pointer of local job size to process
 * @param ranks : ranks participate in this job, nullptr means all ranks
 * @param rank_count : rank count participate in this job, 0 means all ranks
 * @return : offset for local rank
 */
size_t WmmpCollOffsetAndSize(size_t total_size, size_t* local_size, const int* ranks = nullptr, int rank_count = 0);

}