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

#include "whole_memory.h"
#include <iostream>
#include <string>
#include <vector>

namespace whole_graph {

int BCRank(BootstrapCommunicator *bootstrap_communicator);
int BCSize(BootstrapCommunicator *bootstrap_communicator);
void BCAllToAll(const void *send_buf,
                int send_size,
                void *recv_buf,
                int recv_size,
                BootstrapCommunicator *bootstrap_communicator);
void BCAllToAllV(const void **send_bufs,
                 const int *send_sizes,
                 void **recv_bufs,
                 const int *recv_sizes,
                 BootstrapCommunicator *bootstrap_communicator);
void BCBarrier(BootstrapCommunicator *bootstrap_communicator);

void CollBroadcastString(std::string *str, int root, BootstrapCommunicator *bootstrap_communicator);

template<typename T>
inline void CollCheckAllSame(const T &t, BootstrapCommunicator *bc_ptr) {
  int rank = BCRank(bc_ptr);
  int size = BCSize(bc_ptr);
  int rank_count = size;
  std::vector<T> send_buf(rank_count, t);
  std::vector<T> recv_buf(rank_count);
  BCAllToAll(send_buf.data(), sizeof(T), recv_buf.data(), sizeof(T), bc_ptr);
  for (int i = 0; i < rank_count; i++) {
    if (recv_buf[i] != t) {
      std::cerr << "rank " << bc_ptr->Rank() << " data " << recv_buf[i]
                << " different from local rank " << rank << ", data=" << t << std::endl;
      abort();
    }
  }
}

template<typename T>
inline void CollAllGather(const T &t, std::vector<T> *vt, BootstrapCommunicator *bc_ptr) {
  int size = BCSize(bc_ptr);
  int rank_count = size;
  vt->resize(rank_count);
  std::vector<T> send_buf(rank_count, t);
  BCAllToAll(send_buf.data(), sizeof(T), vt->data(), sizeof(T), bc_ptr);
}

template<typename T>
inline T CollBroadcast(const T &t, int root, BootstrapCommunicator *bc_ptr) {
  std::vector<T> vt;
  CollAllGather<T>(t, &vt, bc_ptr);
  return vt[root];
}

}// namespace whole_graph