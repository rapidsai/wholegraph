// nvcc -o cu_mem_handle_perf -lcuda -lcudart cu_mem_handle_perf.cu

#include <assert.h>
#include <cuda.h>
#include <string.h>
#include <sys/time.h>

#include <iostream>
#include <experimental/random>

#define CU_CHECK(X)                                                                           \
do {                                                                                          \
  auto result = X;                                                                            \
  if (result != CUDA_SUCCESS) {                                                               \
    const char* p_err_str = nullptr;                                                          \
    if (cuGetErrorString(result, &p_err_str) == CUDA_ERROR_INVALID_VALUE) {                   \
      p_err_str = "Unrecoginzed CU error num";                                                \
    }                                                                                         \
    std::cerr << "File " << __FILE__ << " Line " << __LINE__ << " " << #X << " returned "     \
              << std::string(p_err_str) << std::endl;                                         \
    abort();                                                                                  \
  }                                                                                           \
} while (0)

#define CUDA_CHECK(X)                                                                         \
do {                                                                                          \
  auto result = X;                                                                            \
  if (result != cudaSuccess) {                                                                \
    const char* p_err_str = cudaGetErrorName(result);                                         \
    std::cerr << "File " << __FILE__ << " Line " << __LINE__ << " " << #X << " returned "     \
              << std::string(p_err_str) << std::endl;                                         \
    abort();                                                                                  \
  }                                                                                           \
} while (0)

#define TIME_DIFF_US(TVS, TVE) ((TVE.tv_sec - TVS.tv_sec)*1000ULL*1000ULL + (TVE.tv_usec - TVS.tv_usec))

__global__ void GatherFloatKernel(float* output, const float* param, int* indice, size_t embedding_dim) {
  int bidx = blockIdx.x;
  int idx = indice[bidx];
  float* output_ptr = (float*)(output + (size_t)bidx * embedding_dim);
  const float* param_ptr = (const float*)(param + (size_t)idx * embedding_dim);
  for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
    output_ptr[i] = param_ptr[i];
  }
}

static constexpr size_t kGatherVecCount = 1024 * 1024;
static constexpr size_t kEmbeddingDim = 128;

void DoMemMapTest(int mem_size_gb, bool export_fd, int grab_mb) {
  size_t mem_size = mem_size_gb * 1024LL * 1024LL * 1024LL;
  assert(mem_size % (kEmbeddingDim * sizeof(float)) == 0);
  size_t vec_count = mem_size / (kEmbeddingDim * sizeof(float));
  float* dst;
  float* src;
  int *idx_h, *idx_d;
  int dev_count;
  CUDA_CHECK(cudaGetDeviceCount(&dev_count));
  assert(dev_count > 0);
  size_t size = kGatherVecCount * kEmbeddingDim * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&dst, size));
  CUDA_CHECK(cudaMalloc((void**)&idx_d, kGatherVecCount * sizeof(int)));
  CUDA_CHECK(cudaMallocHost((void**)&idx_h, kGatherVecCount * sizeof(int)));
  for (size_t i = 0; i < kGatherVecCount; i++) {
    idx_h[i] = std::experimental::randint<int>(0, vec_count - 1);
  }
  CUDA_CHECK(cudaMemcpy(idx_d, idx_h, kGatherVecCount * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  size_t default_granularity, granularity;
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = export_fd ? CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR : CU_MEM_HANDLE_TYPE_NONE;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
  prop.location.id = 0;
  CUmemAllocationGranularity_flags flags = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED;
  CU_CHECK(cuMemGetAllocationGranularity(&default_granularity, &prop, flags));
  granularity = default_granularity;
  size_t requested_granularity = grab_mb * 1024LL * 1024LL;
  if (granularity < requested_granularity && requested_granularity % granularity == 0) {
    granularity = requested_granularity;
  }
  std::cout << "default_granularity=" << default_granularity << ", using granularity=" << granularity << std::endl;
  std::cout << "Using HandleType=" << (export_fd ? "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR" : "CU_MEM_HANDLE_TYPE_NONE") << std::endl;
  size_t size_per_dev = (mem_size + granularity * dev_count - 1) / (granularity * dev_count) * granularity;
  size_t total_alloc_size = size_per_dev * dev_count;

  CUmemGenericAllocationHandle h[dev_count];
  for (int i = 0; i < dev_count; i++) {
    prop.location.id = i;
    CU_CHECK(cuMemCreate(&h[i], size_per_dev, &prop, 0));
  }
  CU_CHECK(cuMemAddressReserve((CUdeviceptr*)&src, total_alloc_size, granularity, 0, 0));
  for (int i = 0; i < dev_count; i++) {
    CU_CHECK(cuMemMap((CUdeviceptr)src + i * size_per_dev, size_per_dev, 0, h[i], 0));
  }
  CUmemAccessDesc madesc;
  madesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  madesc.location.id = 0;
  madesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_CHECK(cuMemSetAccess((CUdeviceptr)src, total_alloc_size, &madesc, 1));

  // Do test.
  // Warm Up
  const int block_size = kEmbeddingDim > 512 ? 512 : kEmbeddingDim;
  GatherFloatKernel<<<1, block_size>>>(dst, src, idx_d, block_size);
  CUDA_CHECK(cudaDeviceSynchronize());

  const int iter_count = 10;
  const int block_count = kGatherVecCount;
  struct timeval tv_s, tv_e;

  gettimeofday(&tv_s, nullptr);
  for (int i = 0; i < iter_count; i++) {
    GatherFloatKernel<<<block_count, block_size>>>(dst, src, idx_d, block_size);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  gettimeofday(&tv_e, nullptr);;
  int time_us = TIME_DIFF_US(tv_s, tv_e);
  size_t total_size = kGatherVecCount * kEmbeddingDim * sizeof(float) * iter_count;
  double bw_gb = total_size / time_us / 1e3;
  std::cout << "Time per iter=" << time_us / iter_count << " us, "
            << "Bandwidth=" << bw_gb << " GB/s" << std::endl;

  for (int i = 0; i < dev_count; i++) {
    CU_CHECK(cuMemUnmap((CUdeviceptr)src + i * size_per_dev, size_per_dev));
    CU_CHECK(cuMemRelease(h[i]));
  }
  CU_CHECK(cuMemAddressFree((CUdeviceptr)src, total_alloc_size));
  CUDA_CHECK(cudaFree(dst));
  CUDA_CHECK(cudaFree(idx_d));
  CUDA_CHECK(cudaFreeHost(idx_h));
}

int main(int argc, char** argv) {
  std::string usage = "Usage: ";
  usage += argv[0];
  usage += " MemSizeGB [F|N] GranularityMB\n\tF is CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, N is CU_MEM_HANDLE_TYPE_NONE.\n";
  if (argc != 4) {
    std::cerr << usage << std::endl;
    return -1;
  }
  int size_GB = atoi(argv[1]);
  std::string handle_type_str = argv[2];
  bool export_fd;
  if (handle_type_str == "F") {
    export_fd = true;
  } else if (handle_type_str == "N") {
    export_fd = false;
  } else {
    std::cerr << "Handle type should be F or N.\n";
    return -1;
  }
  int granularity_MB = atoi(argv[3]);
  CU_CHECK(cuInit(0));
  CUDA_CHECK(cudaSetDevice(0));
  DoMemMapTest(size_GB, export_fd, granularity_MB);
  return 0;
}