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
#include <stdio.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include <vector>

#include "parallel_utils.h"
#include "whole_memory.h"

int RandData(int src_rank, int dst_rank, int nrank, int idx, int seed) {
  return (src_rank * nrank + dst_rank) * 101 + idx * 31 + seed;
}

void AllToAllTest(whole_graph::BootstrapCommunicator *communicator,
                  int elt_count, int rank, int size, int seed) {
  std::vector<int> send_data(elt_count * size);
  for (int r = 0; r < size; r++) {
    for (int idx = 0; idx < elt_count; idx++) {
      send_data[r * elt_count + idx] = RandData(rank, r, size, idx, seed);
    }
  }
  std::vector<int> recv_data(elt_count * size);
  communicator->AllToAll(send_data.data(), elt_count * sizeof(int), recv_data.data(), elt_count * sizeof(int));
  for (int src_r = 0; src_r < size; src_r++) {
    for (int idx = 0; idx < elt_count; idx++) {
      int target = RandData(src_r, rank, size, idx, seed);
      int recv_value = recv_data[src_r * elt_count + idx];
      if (target != recv_value) {
        fprintf(stderr, "Rank=%d, src_rank=%d, recv_value=%d, should be %d\n",
                rank, src_r, recv_value, target);
        abort();
      }
    }
  }
  fprintf(stderr, "Rank=%d AllToAll elt_count=%d passed.\n", rank, elt_count);
}

void AllToAllVTest(whole_graph::BootstrapCommunicator *communicator,
                   int max_elt_count, int rank, int size, int seed) {
  std::vector<std::vector<int>> send_data(size);
  std::vector<std::vector<int>> recv_data(size);
  std::vector<int> send_size(size), recv_size(size);
  std::vector<int *> send_ptrs(size), recv_ptrs(size);
  for (int peer_r = 0; peer_r < size; peer_r++) {
    int send_to_count = (int64_t)(rank * size + (peer_r + seed) % size) * max_elt_count / size / size;
    send_data[peer_r].resize(send_to_count);
    for (int idx = 0; idx < send_to_count; idx++) {
      send_data[peer_r][idx] = RandData(peer_r, rank, size, idx, seed);
    }
    int recv_from_count = (int64_t)(peer_r * size + (rank + seed) % size) * max_elt_count / size / size;
    recv_data[peer_r].resize(recv_from_count);
    send_size[peer_r] = send_to_count * sizeof(int);
    recv_size[peer_r] = recv_from_count * sizeof(int);
    send_ptrs[peer_r] = send_data[peer_r].data();
    recv_ptrs[peer_r] = recv_data[peer_r].data();
  }
  communicator->AllToAllV((const void **) send_ptrs.data(),
                          send_size.data(),
                          (void **) recv_ptrs.data(),
                          recv_size.data());
  for (int src_r = 0; src_r < size; src_r++) {
    int recv_from_count = (int64_t)(src_r * size + (rank + seed) % size) * max_elt_count / size / size;
    for (int idx = 0; idx < recv_from_count; idx++) {
      int target = RandData(rank, src_r, size, idx, seed);
      int recv_value = recv_data[src_r][idx];
      if (target != recv_value) {
        fprintf(stderr, "Rank=%d, src_rank=%d, idx=%d, recv_value=%d, should be %d\n",
                rank, src_r, idx, recv_value, target);
        abort();
      }
    }
  }
  fprintf(stderr, "Rank=%d AllToAllV max_elt_count=%d passed.\n", rank, max_elt_count);
}

void RunAllTests(whole_graph::BootstrapCommunicator *communicator) {
  AllToAllTest(communicator, 1, communicator->Rank(), communicator->Size(), 1);
  AllToAllTest(communicator, 1024 + 1, communicator->Rank(), communicator->Size(), 2);
  AllToAllTest(communicator, 1024 * 1024 - 1, communicator->Rank(), communicator->Size(), 3);
  AllToAllTest(communicator, 1024 * 1024, communicator->Rank(), communicator->Size(), 4);
  AllToAllTest(communicator, 1024 * 1024 + 1, communicator->Rank(), communicator->Size(), 5);

  AllToAllVTest(communicator, 1, communicator->Rank(), communicator->Size(), 1);
  AllToAllVTest(communicator, 1024 + 1, communicator->Rank(), communicator->Size(), 2);
  AllToAllVTest(communicator, 1024 * 1024 - 1, communicator->Rank(), communicator->Size(), 3);
  AllToAllVTest(communicator, 1024 * 1024, communicator->Rank(), communicator->Size(), 4);
  AllToAllVTest(communicator, 1024 * 1024 + 1, communicator->Rank(), communicator->Size(), 5);
}

int main(int argc, char **argv) {
  int dev_count = 8;
  std::vector<std::array<int, 2>> pipes;
  if (CreatePipes(&pipes, dev_count) != 0) {
    return -1;
  }

  MultiProcessRun(dev_count, [&pipes](int rank, int size) {
    if (cudaSetDevice(rank) != cudaSuccess) {
      fprintf(stderr, "Set device failed.\n");
      return;
    }
    auto *communicator = new whole_graph::BootstrapCommunicator();
    whole_graph::WmmpUniqueId unique_id;
    if (rank == 0) {
      whole_graph::WmmpGetUniqueId(&unique_id);
    }
    PipeBroadcast(rank, size, 0, pipes, &unique_id);
    communicator->InitRank(size, unique_id, rank);
    RunAllTests(communicator);
  });

  return 0;
}