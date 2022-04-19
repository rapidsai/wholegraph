#include "whole_memory.h"

namespace whole_memory {

class DomainSocketCommunicatorImpl;

class DomainSocketCommunicator : public CollectiveCommunicator {
 public:
  DomainSocketCommunicator();
  ~DomainSocketCommunicator() override;
  /*!
   * Set Rank and Size for Communicator.
   * NOTE this is not world rank and size, but rank and size inside WholeMemory Group.
   *   e.g. process inside one DGX-A100 node may be one WholeMemory Group
   * @param rank : group rank of current process
   * @param size : communication group size for current process.
   */
  void SetRankAndSize(int rank, int size) override;
  /*!
   * Get Rank of current process.
   * @return rank
   */
  int Rank() override;
  /*!
   * Get Size of collective processes.
   * @return size
   */
  int Size() override;
  /*!
   * AllToAll with ranks specified
   * @param send_buf : send data buffer
   * @param send_size : send size to each rank
   * @param recv_buf : receive data buffer
   * @param recv_size : receive size from each rank
   * @param ranks : ranks participate in this AllToAll, nullptr means all ranks
   * @param rank_count : rank count participate in this AllToAll, 0 means all ranks
   */
  void AllToAll(const void* send_buf, int send_size, void* recv_buf, int recv_size, const int* ranks, int rank_count) override;
  /*!
   * AllToAllV with ranks specified
   * @param send_bufs : send data buffers
   * @param send_sizes : send sizes to each rank
   * @param recv_bufs : receive data buffers
   * @param recv_sizes : receive sizes from each rank
   * @param ranks : ranks participate in this AllToAll, nullptr means all ranks
   * @param rank_count : rank count participate in this AllToAll, 0 means all ranks
   */
  void AllToAllV(const void** send_bufs, const int* send_sizes, void** recv_bufs, const int* recv_sizes, const int* ranks, int rank_count) override;

  DomainSocketCommunicatorImpl* impl_ = nullptr;
};

}