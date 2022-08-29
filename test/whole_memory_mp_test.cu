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
#include "whole_memory_test_utils.cuh"

void main_func(int rank, int size, const std::vector<std::array<int, 2>> &pipes) {
  whole_graph::WholeMemoryInit();
  int dev_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&dev_count));
  CUDA_CHECK(cudaSetDevice(rank % dev_count));

  whole_graph::WmmpUniqueId a_unique_id;
  if (rank == 0) {
    whole_graph::WmmpGetUniqueId(&a_unique_id);
  }
  PipeBroadcast(rank, size, 0, pipes, &a_unique_id);
  auto *a_bc_ptr = whole_graph::WmmpCreateCommunicator(size, a_unique_id, rank);

  whole_graph::WmmpUniqueId b_unique_id;
  if (rank % 2 == 0) {
    whole_graph::WmmpGetUniqueId(&b_unique_id);
  }
  PipeGroupBroadcast(rank, size, 0, 2, pipes, &b_unique_id);
  auto *b_bc_ptr = whole_graph::WmmpCreateCommunicator(size / 4, b_unique_id, rank % 2);

  whole_graph::WmmpBarrier(a_bc_ptr);

  whole_graph::WmmpUniqueId c_unique_id;
  if (rank % 4 == 0) {
    whole_graph::WmmpGetUniqueId(&c_unique_id);
  }
  PipeGroupBroadcast(rank, size, 0, 4, pipes, &c_unique_id);
  auto *c_bc_ptr = whole_graph::WmmpCreateCommunicator(size / 2, c_unique_id, rank % 4);
  whole_graph::WmmpBarrier(a_bc_ptr);

  const int kEltCount = 1000 * 1000 * 200;

  std::vector<int> bv, cv;
  int b_start = rank / 2 * 2;
  int c_start = rank / 4 * 4;
  bv.resize(2, b_start);
  bv[1] = b_start + 1;
  cv.resize(4, c_start);
  for (int i = 0; i < 4; i++) cv[i] = c_start + i;

  int *a, *b, *c;
  int *a_h, *b_h, *c_h;
  whole_graph::WmmpMalloc((void **) &a, kEltCount * sizeof(int), a_bc_ptr);
  whole_graph::WmmpMalloc((void **) &b, kEltCount * sizeof(int), b_bc_ptr);
  whole_graph::WmmpMalloc((void **) &c, kEltCount * sizeof(int), c_bc_ptr);
  whole_graph::WmmpMallocHost((void **) &a_h, kEltCount * sizeof(int), a_bc_ptr);
  whole_graph::WmmpMallocHost((void **) &b_h, kEltCount * sizeof(int), b_bc_ptr);
  whole_graph::WmmpMallocHost((void **) &c_h, kEltCount * sizeof(int), c_bc_ptr);

  if (rank == 0) {
    WriteData(a, kEltCount, 0);
    WriteDataCPU(a_h, kEltCount, 1000);
  } else if (rank == 1 || rank == 3 || rank == 5 || rank == 7) {
    WriteData(b, kEltCount, 1234 + rank);
    WriteData(b_h, kEltCount, 2234 + rank);
  } else if (rank == 2 || rank == 6) {
    WriteData(c, kEltCount, 3323 + rank);
    WriteDataCPU(c_h, kEltCount, 4323 + rank);
  }

  whole_graph::WmmpBarrier(a_bc_ptr);

  CheckData(a, kEltCount, 0);
  fprintf(stderr, "Rank %d done Check data for a.\n", rank);
  CheckData(b, kEltCount, 1234 + rank / 2 * 2 + 1);
  fprintf(stderr, "Rank %d done Check data for b.\n", rank);
  CheckData(c, kEltCount, 3323 + rank / 4 * 4 + 2);
  fprintf(stderr, "Rank %d done Check data for c.\n", rank);

  CheckData(a_h, kEltCount, 1000);
  fprintf(stderr, "Rank %d done Check data for a_h.\n", rank);
  CheckData(b_h, kEltCount, 2234 + rank / 2 * 2 + 1);
  fprintf(stderr, "Rank %d done Check data for b_h.\n", rank);
  CheckData(c_h, kEltCount, 4323 + rank / 4 * 4 + 2);
  fprintf(stderr, "Rank %d done Check data for c_h.\n", rank);

  fprintf(stderr, "Rank %d Start free b.\n", rank);
  whole_graph::WmmpFree(b);
  fprintf(stderr, "Rank %d After free b.\n", rank);
  whole_graph::WmmpFree(c);
  fprintf(stderr, "Rank %d After free c.\n", rank);
  whole_graph::WmmpFree(c_h);
  fprintf(stderr, "Rank %d After free c_h.\n", rank);
  whole_graph::WmmpFree(a_h);
  fprintf(stderr, "Rank %d After free a_h.\n", rank);

  whole_graph::WmmpDestroyCommunicator(a_bc_ptr);
  whole_graph::WmmpDestroyCommunicator(b_bc_ptr);
  whole_graph::WmmpDestroyCommunicator(c_bc_ptr);

  whole_graph::WholeMemoryFinalize();
}

int main(int argc, char **argv) {
  int dev_count = ForkGetDeviceCount();
  fprintf(stderr, "Device count=%d\n", dev_count);
  if (dev_count < 8) {
    fprintf(stderr, "Need 8 GPUs to run.\n");
    return -1;
  }
  std::vector<std::array<int, 2>> pipes;
  if (CreatePipes(&pipes, dev_count) != 0) {
    return -1;
  }
  MultiProcessRun(dev_count, [&pipes](int rank, int size) {
    main_func(rank, size, pipes);
  });
  return 0;
}