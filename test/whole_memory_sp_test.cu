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
#include <cuda_runtime_api.h>
#include <stdio.h>

#include <vector>

#include "parallel_utils.h"
#include "whole_memory.h"
#include "whole_memory_test_utils.cuh"

#define TEST_GPU_COUNT (8)

int main(int argc, char **argv) {
  whole_graph::WholeMemoryInit();

  const int kEltCount = 1000 * 1000 * 200;
  int dev_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&dev_count));

  std::vector<int> av, bv, cv;
  av.resize(8, 0);
  for (int i = 0; i < 8; i++) av[i] = i % dev_count;
  bv.resize(2, 2);
  bv[1] = 1;
  cv.resize(4, 4);
  for (int i = 0; i < 4; i++) cv[i] = (4 + i) % dev_count;

  int *a, *b, *c;
  int *a_h, *b_h, *c_h;
  whole_graph::WmspMalloc((void **) &a, kEltCount * sizeof(int), av.data(), av.size());
  fprintf(stderr, "Allocated a (all)\n");
  //getchar();
  whole_graph::WmspMalloc((void **) &b, kEltCount * sizeof(int), bv.data(), bv.size());
  fprintf(stderr, "Allocated b (2,1)\n");
  //getchar();
  whole_graph::WmspMalloc((void **) &c, kEltCount * sizeof(int), cv.data(), cv.size());
  fprintf(stderr, "Allocated c (4,5,6,7)\n");
  //getchar();

  whole_graph::WmspMallocHost((void **) &a_h, kEltCount * sizeof(int));
  fprintf(stderr, "Allocated a_h\n");
  //getchar();
  whole_graph::WmspMallocHost((void **) &b_h, kEltCount * sizeof(int));
  fprintf(stderr, "Allocated b_h\n");
  //getchar();
  whole_graph::WmspMallocHost((void **) &c_h, kEltCount * sizeof(int));
  fprintf(stderr, "Allocated c_h\n");
  //getchar();

  MultiThreadRun(TEST_GPU_COUNT, [kEltCount, a, b, c, a_h, b_h, c_h, dev_count](int rank, int size) {
    CUDA_CHECK(cudaSetDevice(rank % dev_count));
    if (rank == 5) {
      WriteData(a, kEltCount, 0);
    } else if (rank == 2) {
      WriteData(b, kEltCount, 1234);
    } else if (rank == 7) {
      WriteData(c, kEltCount, 3323);
    } else if (rank == 4) {
      WriteDataCPU(a_h, kEltCount, 1000);
    } else if (rank == 0) {
      WriteData(b_h, kEltCount, 2234);
    } else if (rank == 6) {
      WriteDataCPU(c_h, kEltCount, 4323);
    }
  });
  fprintf(stderr, "Done Writing data.\n");

  MultiThreadRun(TEST_GPU_COUNT, [kEltCount, a, b, c, a_h, b_h, c_h, dev_count](int rank, int size) {
    CUDA_CHECK(cudaSetDevice(rank % dev_count));
    if (rank == 1) {
      CheckData(b, kEltCount, 1234);
      fprintf(stderr, "Rank %d done Check data for b.\n", rank);
    } else if (rank == 6) {
      CheckData(c, kEltCount, 3323);
      fprintf(stderr, "Rank %d done Check data for c.\n", rank);
    } else if (rank == 3) {
      CheckData(a_h, kEltCount, 1000);
      fprintf(stderr, "Rank %d done Check data for a_h.\n", rank);
    } else if (rank == 2) {
      CheckDataCPU(b_h, kEltCount, 2234);
      fprintf(stderr, "Rank %d done Check data for b_h.\n", rank);
    } else if (rank == 5) {
      CheckData(c_h, kEltCount, 4323);
      fprintf(stderr, "Rank %d done Check data for c_h.\n", rank);
    } else {
      CheckData(a, kEltCount, 0);
      fprintf(stderr, "Rank %d done Check data for a.\n", rank);
    }
  });
  fprintf(stderr, "Done check data.\n");

  fprintf(stderr, "Before free.\n");
  //getchar();
  whole_graph::WmspFree(b);
  fprintf(stderr, "Freed b.\n");
  //getchar();
  whole_graph::WmspFree(c);
  fprintf(stderr, "Freed c.\n");
  //getchar();
  whole_graph::WmspFree(c_h);
  fprintf(stderr, "Freed c_h.\n");
  //getchar();
  whole_graph::WmspFree(a);
  fprintf(stderr, "Freed a_h.\n");
  //getchar();

  whole_graph::WholeMemoryFinalize();
  return 0;
}