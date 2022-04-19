#include <assert.h>
#include <stdio.h>
#include <cuda_runtime_api.h>

#include <vector>

#include "whole_memory.h"
#include "whole_memory_test_utils.cuh"
#include "parallel_utils.h"

#define TEST_GPU_COUNT (8)

int main(int argc, char** argv) {
  whole_memory::WholeMemoryInit();

  const int kEltCount = 1000 * 1000 * 200;
  int dev_count = 0;
  assert(cudaGetDeviceCount(&dev_count) == cudaSuccess);

  std::vector<int> av, bv, cv;
  av.resize(8, 0);
  for (int i = 0; i < 8; i++) av[i] = i % dev_count;
  bv.resize(2, 2);
  bv[1] = 1;
  cv.resize(4, 4);
  for (int i = 0; i < 4; i++) cv[i] = (4 + i) % dev_count;

  int *a, *b, *c;
  int *a_h, *b_h, *c_h;
  whole_memory::WmspMalloc((void**)&a, kEltCount * sizeof(int), av.data(), av.size());
  std::cout << "Allocated a (all)\n";
  //getchar();
  whole_memory::WmspMalloc((void**)&b, kEltCount * sizeof(int), bv.data(), bv.size());
  std::cout << "Allocated b (2,1)\n";
  //getchar();
  whole_memory::WmspMalloc((void**)&c, kEltCount * sizeof(int), cv.data(), cv.size());
  std::cout << "Allocated c (4,5,6,7)\n";
  //getchar();

  whole_memory::WmspMallocHost((void**)&a_h, kEltCount * sizeof(int));
  std::cout << "Allocated a_h\n";
  //getchar();
  whole_memory::WmspMallocHost((void**)&b_h, kEltCount * sizeof(int));
  std::cout << "Allocated b_h\n";
  //getchar();
  whole_memory::WmspMallocHost((void**)&c_h, kEltCount * sizeof(int));
  std::cout << "Allocated c_h\n";
  //getchar();

  MultiThreadRun(TEST_GPU_COUNT, [kEltCount, a, b, c, a_h, b_h, c_h, dev_count](int rank, int size) {
    assert(cudaSetDevice(rank % dev_count) == cudaSuccess);
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
  std::cout << "Done Writing data." << std::endl;

  MultiThreadRun(TEST_GPU_COUNT, [kEltCount, a, b, c, a_h, b_h, c_h, dev_count](int rank, int size) {
    assert(cudaSetDevice(rank % dev_count) == cudaSuccess);
    if (rank == 1) {
      CheckData(b, kEltCount, 1234);
      std::cout << "Rank " << rank << " done Check data for b." << std::endl;
    } else if (rank == 6) {
      CheckData(c, kEltCount, 3323);
      std::cout << "Rank " << rank << " done Check data for c." << std::endl;
    } else if (rank == 3) {
      CheckData(a_h, kEltCount, 1000);
      std::cout << "Rank " << rank << " done Check data for a_h." << std::endl;
    } else if (rank == 2) {
      CheckDataCPU(b_h, kEltCount, 2234);
      std::cout << "Rank " << rank << " done Check data for b_h." << std::endl;
    } else if (rank == 5) {
      CheckData(c_h, kEltCount, 4323);
      std::cout << "Rank " << rank << " done Check data for c_h." << std::endl;
    } else {
      CheckData(a, kEltCount, 0);
      std::cout << "Rank " << rank << " done Check data for a." << std::endl;
    }
  });
  std::cout << "Done Checking data." << std::endl;

  std::cout << "Before free\n";
  //getchar();
  whole_memory::WmspFree(b);
  std::cout << "Freed b\n";
  //getchar();
  whole_memory::WmspFree(c);
  std::cout << "Freed c\n";
  //getchar();
  whole_memory::WmspFree(c_h);
  std::cout << "Freed c_h\n";
  //getchar();
  whole_memory::WmspFree(a);
  std::cout << "Freed a_h\n";
  //getchar();

  whole_memory::WholeMemoryFinalize();
  return 0;
}