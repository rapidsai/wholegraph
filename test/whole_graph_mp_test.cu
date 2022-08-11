#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include <vector>

#include "whole_graph.h"
#include "whole_graph_test_utils.cuh"
#include "parallel_utils.h"

void main_func(int rank, int size) {
  whole_graph::WholeMemoryInit();
  int dev_count = 0;
  assert(cudaGetDeviceCount(&dev_count) == cudaSuccess);
  assert(cudaSetDevice(rank % dev_count) == cudaSuccess);
  whole_graph::WmmpInit(rank, size, nullptr);

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
  whole_graph::WmmpMalloc((void**)&a, kEltCount * sizeof(int));
  whole_graph::WmmpMalloc((void**)&b, kEltCount * sizeof(int), bv.data(), 2);
  whole_graph::WmmpMalloc((void**)&c, kEltCount * sizeof(int), cv.data(), 4);
  whole_graph::WmmpMallocHost((void**)&a_h, kEltCount * sizeof(int));
  whole_graph::WmmpMallocHost((void**)&b_h, kEltCount * sizeof(int), bv.data(), 2);
  whole_graph::WmmpMallocHost((void**)&c_h, kEltCount * sizeof(int), cv.data(), 4);

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

  whole_graph::WmmpBarrier();

  CheckData(a, kEltCount, 0);
  std::cout << "Rank " << rank << " done Check data for a." << std::endl;
  CheckData(b, kEltCount, 1234 + rank / 2 * 2 + 1);
  std::cout << "Rank " << rank << " done Check data for b." << std::endl;
  CheckData(c, kEltCount, 3323 + rank / 4 * 4 + 2);
  std::cout << "Rank " << rank << " done Check data for c." << std::endl;

  CheckData(a_h, kEltCount, 1000);
  std::cout << "Rank " << rank << " done Check data for a_h." << std::endl;
  CheckData(b_h, kEltCount, 2234 + rank / 2 * 2 + 1);
  std::cout << "Rank " << rank << " done Check data for b_h." << std::endl;
  CheckData(c_h, kEltCount, 4323 + rank / 4 * 4 + 2);
  std::cout << "Rank " << rank << " done Check data for c_h." << std::endl;

  std::cerr << "Start free b" << std::endl;
  whole_graph::WmmpFree(b);
  std::cerr << "After free b" << std::endl;
  whole_graph::WmmpFree(c);
  std::cerr << "After free c" << std::endl;
  whole_graph::WmmpFree(c_h);
  std::cerr << "After free c_h" << std::endl;
  whole_graph::WmmpFree(a_h);
  std::cerr << "After free a_h" << std::endl;

  whole_graph::WholeMemoryFinalize();
}

int main(int argc, char** argv) {
  MultiProcessRun(8, [](int rank, int size){
    main_func(rank, size);
  });
  return 0;
}