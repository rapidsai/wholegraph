/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>

#include <cuda_runtime_api.h>
#include <sys/mman.h>

#include <atomic>

#include "parallel_utils.hpp"

TEST(ParallelUtilsTest, MultiThreadRun)
{
  std::atomic<int> thread_count = 0;
  std::atomic<int> thread_mask  = 0;
  MultiThreadRun(8, [&thread_count, &thread_mask](int rank, int size) {
    thread_count.fetch_add(1);
    thread_mask.fetch_or(1 << rank);
    EXPECT_EQ(size, 8);
  });
  EXPECT_EQ(thread_count.load(), 8);
  EXPECT_EQ(thread_mask.load(), (1 << 8) - 1);
}

TEST(ParallelUtilsTest, MultiProcessRun)
{
  int* shared_info = static_cast<int*>(
    mmap(nullptr, sizeof(int) * 2, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  int* thread_count = shared_info;
  int* thread_mask  = shared_info + 1;
  *thread_count     = 0;
  *thread_mask      = 0;
  MultiProcessRun(8, [thread_count, thread_mask](int rank, int world_size) {
    reinterpret_cast<std::atomic<int>*>(thread_count)->fetch_add(1);
    reinterpret_cast<std::atomic<int>*>(thread_mask)->fetch_or(1 << rank);
    EXPECT_EQ(world_size, 8);
    // need to manually check gtest failures and modify exit code, WHOLEMEMORY_CHECK can help do
    // thisl
    WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
  });
  EXPECT_EQ(*thread_count, 8);
  EXPECT_EQ(*thread_mask, (1 << 8) - 1);
}

TEST(ParallelUtilsTest, PipeBroadcast)
{
  const int nproc = 8;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, nproc);
  for (int root = 0; root < nproc; root++) {
    MultiProcessRun(nproc, [root, &pipes](int rank, int world_size) {
      int data = rank * 10;
      PipeBroadcast(rank, world_size, root, pipes, &data);
      EXPECT_EQ(data, root * 10);
      // need to manually check gtest failures and modify exit code, WHOLEMEMORY_CHECK can help do
      // thisl
      WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
    });
  }
  ClosePipes(&pipes);
}

TEST(ParallelUtilsTest, GroupPipeBroadcast)
{
  const int nproc = 8;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, nproc);
  const int group_size = 4;
  for (int group_root = 0; group_root < group_size; group_root++) {
    MultiProcessRun(nproc, [group_root, &pipes](int rank, int world_size) {
      int data = rank * 10;
      PipeGroupBroadcast(rank, world_size, group_root, group_size, pipes, &data);
      int rank_group_root = group_root + rank / group_size * group_size;
      EXPECT_EQ(data, rank_group_root * 10);
      // need to manually check gtest failures and modify exit code, WHOLEMEMORY_CHECK can help do
      // thisl
      WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
    });
  }
  ClosePipes(&pipes);
}

TEST(ParallelUtilsTest, ForkGetDeviceCount)
{
  int dev_count_fork = ForkGetDeviceCount();
  int dev_count_cuda;
  EXPECT_EQ(cudaGetDeviceCount(&dev_count_cuda), cudaSuccess);
  EXPECT_EQ(dev_count_cuda, dev_count_fork);
}
