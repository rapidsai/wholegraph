#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace whole_memory {

int CCRank();
int CCSize();
void CCAllToAll(const void* send_buf, int send_size, void* recv_buf, int recv_size, const int* ranks, int rank_count);
void CCAllToAllV(const void** send_bufs, const int* send_sizes, void** recv_bufs, const int* recv_sizes, const int* ranks, int rank_count);
void CCBarrier(const int* ranks, int rank_count);

void CheckRanksValid(const int* ranks, int rank_count);
int GetRankOffsetInRanks(int rank, const int* ranks, int rank_count);
int GetSizeInRanks(const int* ranks, int rank_count);
void CollBroadcastString(std::string* str, int root, const int* ranks, int rank_count);

template <typename T>
inline void CollCheckAllSame(const T& t, const int* ranks, int rank_count) {
  int rank = CCRank();
  int size = CCSize();
  if (ranks == nullptr || rank_count == 0) rank_count = size;
  std::vector<T> send_buf(rank_count, t);
  std::vector<T> recv_buf(rank_count);
  CCAllToAll(send_buf.data(), sizeof(T), recv_buf.data(), sizeof(T), ranks, rank_count);
  for (int i = 0; i < rank_count; i++) {
    if (recv_buf[i] != t) {
      std::cerr << "rank " << ((ranks == nullptr || rank_count == 0) ? i : ranks[i]) << " data " << recv_buf[i]
                << " different from local rank " << rank << ", data=" << t << std::endl;
      abort();
    }
  }
}

template <typename T>
inline void CollAllGather(const T& t, std::vector<T>* vt, const int* ranks, int rank_count) {
  int size = CCSize();
  if (ranks == nullptr || rank_count == 0) rank_count = size;
  vt->resize(rank_count);
  std::vector<T> send_buf(rank_count, t);
  CCAllToAll(send_buf.data(), sizeof(T), vt->data(), sizeof(T), ranks, rank_count);
}

template <typename T>
inline T CollBroadcast(const T& t, int root, const int* ranks, int rank_count) {
  std::vector<T> vt;
  CollAllGather<T>(t, &vt, ranks, rank_count);
  return vt[root];
}



}