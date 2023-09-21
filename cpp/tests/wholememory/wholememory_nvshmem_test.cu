#include <gtest/gtest.h>

#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"

#include "nvshmem.h"
#include "nvshmemx.h"
#include "wholememory/nvshmem_template.cuh"
#include "wholememory_test_utils.hpp"

#include <wholememory/device_reference.cuh>

__global__ void simple_shift(int* destination)
{
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;

  nvshmem_int_p(destination, mype, peer);
}

__global__ void simple_shift_p2p(int* destination)
{
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;

  // nvshmem_int_p(destination, mype, peer);
  int* dest = static_cast<int*>(nvshmem_ptr(destination, peer));
  dest[0]   = mype;
}

__global__ void read_next_rank_data(
  int* output, wholememory_gref_t global_tensor_ptr, int all_size_of_data, int rank, int world_size)
{
  size_t next_rank              = (rank + 1) % world_size;
  size_t eles_num_each_rank     = (all_size_of_data + world_size - 1) / world_size;
  size_t start_offest_next_rank = eles_num_each_rank * next_rank;
  size_t end_next_rank          = min<int>(eles_num_each_rank * (next_rank + 1), all_size_of_data);
  int eles_this_rank            = end_next_rank - start_offest_next_rank;
// printf("*****************in kernel  read_next_rank_data *************\n");
  const int stride = gridDim.x * blockDim.x;

  wholememory::device_reference<int> global_device_ref{global_tensor_ptr};

  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < eles_this_rank; id += stride) {
    // output[id]=global_device_ref[start_offest_this_rank+id];
    output[id] = global_device_ref.load(start_offest_next_rank + id);
  }
}

void copy_host_array_to_wholememory_v2(void* host_array,
                                       wholememory_handle_t array_handle,
                                       wholememory_array_description_t array_desc,
                                       cudaStream_t stream)
{
  void* local_array_ptr;
  size_t local_array_size, local_array_offset;
  EXPECT_EQ(wholememory_get_local_memory(
              &local_array_ptr, &local_array_size, &local_array_offset, array_handle),
            WHOLEMEMORY_SUCCESS);
  int64_t array_ele_size = wholememory_dtype_get_element_size(array_desc.dtype);
  EXPECT_EQ(local_array_size % array_ele_size, 0);
  EXPECT_EQ(local_array_offset % array_ele_size, 0);
  wholememory_comm_t wm_comm;
  EXPECT_EQ(wholememory_get_communicator(&wm_comm, array_handle), WHOLEMEMORY_SUCCESS);
  printf("****************local_array_ptr :%ld ,local_array_size : %ld*********************\n",
         local_array_ptr,
         local_array_size);
  if (local_array_size) {
    EXPECT_EQ(cudaMemcpyAsync(local_array_ptr,
                              static_cast<char*>(host_array) + local_array_offset,
                              local_array_size,
                              cudaMemcpyHostToDevice,
                              stream),
              cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
  wholememory_communicator_barrier(wm_comm);
}

TEST(WholeMemoryNvshmemTest, SIMPLE_SHIFT)
{
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEMEMORY_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(nproc, [&pipes](int rank, int world_size) {
    // setenv("NVSHMEM_BOOTSTRAP","plugin",1);
    // setenv("NVSHMEM_BOOTSTRAP_PLUGIN","libnvshmem_wholememory_bootstrap.so",1);
    setenv("NVSHMEM_BOOTSTRAP", "mpi", 1);
    setenv("NVSHMEM_BOOTSTRAP_MPI_PLUGIN", "libnvshmem_wholememory_bootstrap.so", 1);
    EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);

    wholememory_comm_t wm_comm1 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wm_comm1->comm_id, 0);

    nvshmemx_init_attr_t attr;
    int mype, npes, mype_node;

    attr.mpi_comm = &wm_comm1;

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    mype      = nvshmem_my_pe();
    npes      = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    EXPECT_EQ(mype, rank);
    EXPECT_EQ(npes, world_size);

    EXPECT_EQ(mype_node, rank);
    // EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);
    cudaStream_t stream;

    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    int* destination = (int*)nvshmem_malloc(sizeof(int));
    EXPECT_NE(destination, nullptr);
    simple_shift<<<1, 1, 0, stream>>>(destination);
    nvshmemx_barrier_all_on_stream(stream);
    int msg;
    EXPECT_EQ(cudaMemcpyAsync(&msg, destination, sizeof(int), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    printf("%d: received message %d\n", nvshmem_my_pe(), msg);

    int expect_msg = (mype + npes - 1) % npes;
    EXPECT_EQ(msg, expect_msg);
    nvshmem_free(destination);
    nvshmem_finalize();
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

    WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
  });
}

TEST(WholeMemoryNvshmemTest, SIMPLE_SHIFT_P2P)
{
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEMEMORY_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(nproc, [&pipes](int rank, int world_size) {
    // setenv("NVSHMEM_BOOTSTRAP","plugin",1);
    // setenv("NVSHMEM_BOOTSTRAP_PLUGIN","libnvshmem_wholememory_bootstrap.so",1);
    setenv("NVSHMEM_BOOTSTRAP", "mpi", 1);
    setenv("NVSHMEM_BOOTSTRAP_MPI_PLUGIN", "libnvshmem_wholememory_bootstrap.so", 1);
    EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);

    wholememory_comm_t wm_comm1 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wm_comm1->comm_id, 0);

    nvshmemx_init_attr_t attr;
    int mype, npes, mype_node;

    attr.mpi_comm = &wm_comm1;

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    mype      = nvshmem_my_pe();
    npes      = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    EXPECT_EQ(mype, rank);
    EXPECT_EQ(npes, world_size);

    EXPECT_EQ(mype_node, rank);
    // EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);
    cudaStream_t stream;

    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    int* destination = (int*)nvshmem_malloc(sizeof(int));
    EXPECT_NE(destination, nullptr);
    simple_shift_p2p<<<1, 1, 0, stream>>>(destination);
    nvshmemx_barrier_all_on_stream(stream);
    int msg;
    EXPECT_EQ(cudaMemcpyAsync(&msg, destination, sizeof(int), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    printf("%d: received message %d\n", nvshmem_my_pe(), msg);

    int expect_msg = (mype + npes - 1) % npes;
    EXPECT_EQ(msg, expect_msg);
    nvshmem_free(destination);
    nvshmem_finalize();
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

    WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
  });
}

TEST(WholeMemoryNvshmemTest, NVSHMEM_TENSOR_GET)
{
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEMEMORY_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(nproc, [&pipes](int rank, int world_size) {


    // setenv("NVSHMEM_BOOTSTRAP","plugin",1);
    // setenv("NVSHMEM_BOOTSTRAP_PLUGIN","libnvshmem_wholememory_bootstrap.so",1);
    setenv("NVSHMEM_BOOTSTRAP", "mpi", 1);
    setenv("NVSHMEM_BOOTSTRAP_MPI_PLUGIN", "libnvshmem_wholememory_bootstrap.so", 1);
    EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);

    wholememory_comm_t wm_comm1 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wm_comm1->comm_id, 0);

    nvshmemx_init_attr_t attr;
    int mype, npes, mype_node;

    attr.mpi_comm = &wm_comm1;

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    // TODO: run in multi node
    mype      = nvshmem_my_pe();
    npes      = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    EXPECT_EQ(mype, rank);
    EXPECT_EQ(npes, world_size);

    EXPECT_EQ(mype_node, rank);
    // EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);
    cudaStream_t stream;
    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    size_t tensor_size = 8192;
    EXPECT_EQ(tensor_size % world_size, 0);
    wholememory_dtype_t data_type = WHOLEMEMORY_DT_INT;
    auto memory_type              = WHOLEMEMORY_MT_NVSHMEM;
    auto memory_location          = WHOLEMEMORY_ML_DEVICE;

    auto array_desc = wholememory_create_array_desc(tensor_size, 0, data_type);

    wholememory_handle_t tensor_memory_handle;
    EXPECT_EQ(wholememory_malloc(&tensor_memory_handle,
                                 wholememory_get_memory_size_from_array(&array_desc),
                                 wm_comm1,
                                 memory_type,
                                 memory_location,
                                 wholememory_dtype_get_element_size(data_type)),
              WHOLEMEMORY_SUCCESS);
    EXPECT_NE(tensor_memory_handle, nullptr);
    std::vector<int> host_vec(tensor_size);
    size_t elements_each_rank = tensor_size / world_size;

    constexpr int OFFSET=100000;
    for (int i = 0; i < tensor_size; i++) {
      host_vec[i] = (i / elements_each_rank)*OFFSET+i;
    }

    printf("********************* tensor_memory_handle:%ld*************\n", tensor_memory_handle);
    copy_host_array_to_wholememory_v2(host_vec.data(), tensor_memory_handle, array_desc, stream);

    wholememory_gref_t wm_tensor_ptr_gref;

    EXPECT_EQ(wholememory_get_global_reference(&wm_tensor_ptr_gref, tensor_memory_handle),
              WHOLEMEMORY_SUCCESS);

    int len_this_rank = (tensor_size + world_size - 1) / world_size;
    int* output_d;
    EXPECT_EQ(cudaMalloc(&output_d, sizeof(int) * len_this_rank), cudaSuccess);
    read_next_rank_data<<<1, 256, 0, stream>>>(
      output_d, wm_tensor_ptr_gref, tensor_size, rank, world_size);

    int next_rank = (rank + 1) % world_size;

    std::vector<int> host_ref(len_this_rank);

    EXPECT_EQ(
      cudaMemcpyAsync(
        host_ref.data(), output_d, sizeof(int) * len_this_rank, cudaMemcpyDeviceToHost, stream),
      cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int i = 0; i < len_this_rank; i++) {
      EXPECT_EQ(host_ref[i], (next_rank*OFFSET+next_rank*len_this_rank+i));
    }

    EXPECT_EQ(wholememory_free(tensor_memory_handle), WHOLEMEMORY_SUCCESS);

    nvshmem_finalize();
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

    WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
  });
}
// ./gtests/WHOLEMEMORY_NVSHMEM_TEST --gtest_filter=*NVSHMEM_TENSOR_GET*



TEST(WholeMemoryNvshmemTest, NVSHMEM_TENSOR_GET_WITH_COMM)
{
  int dev_count = ForkGetDeviceCount();
  // int dev_count=1;
  EXPECT_GE(dev_count, 1);
  WHOLEMEMORY_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(nproc, [&pipes](int rank, int world_size) {

    
    EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);

    wholememory_comm_t wm_comm1 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wm_comm1->comm_id, 0);

  
    // EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);
    cudaStream_t stream;
    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    size_t tensor_size = 8192;
    EXPECT_EQ(tensor_size % world_size, 0);
    wholememory_dtype_t data_type = WHOLEMEMORY_DT_INT;
    auto memory_type              = WHOLEMEMORY_MT_NVSHMEM;
    auto memory_location          = WHOLEMEMORY_ML_DEVICE;

    auto array_desc = wholememory_create_array_desc(tensor_size, 0, data_type);

    wholememory_handle_t tensor_memory_handle;
    EXPECT_EQ(wholememory_malloc(&tensor_memory_handle,
                                 wholememory_get_memory_size_from_array(&array_desc),
                                 wm_comm1,
                                 memory_type,
                                 memory_location,
                                 wholememory_dtype_get_element_size(data_type)),
              WHOLEMEMORY_SUCCESS);
    EXPECT_NE(tensor_memory_handle, nullptr);
    std::vector<int> host_vec(tensor_size);
    size_t elements_each_rank = tensor_size / world_size;

    constexpr int OFFSET=100000;
    for (int i = 0; i < tensor_size; i++) {
      host_vec[i] = (i / elements_each_rank)*OFFSET+i;
    }

    printf("********************* tensor_memory_handle:%ld*************\n", tensor_memory_handle);
    copy_host_array_to_wholememory_v2(host_vec.data(), tensor_memory_handle, array_desc, stream);

    wholememory_gref_t wm_tensor_ptr_gref;

    EXPECT_EQ(wholememory_get_global_reference(&wm_tensor_ptr_gref, tensor_memory_handle),
              WHOLEMEMORY_SUCCESS);

    int len_this_rank = (tensor_size + world_size - 1) / world_size;
    int* output_d;
    EXPECT_EQ(cudaMalloc(&output_d, sizeof(int) * len_this_rank), cudaSuccess);
    read_next_rank_data<<<1, 256, 0, stream>>>(
      output_d, wm_tensor_ptr_gref, tensor_size, rank, world_size);
    
    int next_rank = (rank + 1) % world_size;

    std::vector<int> host_ref(len_this_rank);
    WM_CUDA_CHECK(cudaGetLastError());
    WM_CUDA_CHECK(
      cudaMemcpyAsync(
        host_ref.data(), output_d, sizeof(int) * len_this_rank, cudaMemcpyDeviceToHost, stream));
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int i = 0; i < len_this_rank; i++) {
      EXPECT_EQ(host_ref[i], (next_rank*OFFSET+next_rank*len_this_rank+i));
    }

    EXPECT_EQ(wholememory_free(tensor_memory_handle), WHOLEMEMORY_SUCCESS);

    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

    WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
  });
}

// ./gtests/WHOLEMEMORY_NVSHMEM_TEST --gtest_filter=*NVSHMEM_TENSOR_GET_WITH_COMM*
