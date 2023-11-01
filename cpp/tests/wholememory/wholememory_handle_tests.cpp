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

#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/initialize.hpp"
#include "wholememory/memory_handle.hpp"

#include "wholememory_test_utils.hpp"

class WholeMemoryHandleCreateDestroyParameterTests
  : public ::testing::TestWithParam<
      std::tuple<size_t, wholememory_memory_type_t, wholememory_memory_location_t, size_t>> {};

TEST_P(WholeMemoryHandleCreateDestroyParameterTests, CreateDestroyTest)
{
  auto params   = GetParam();
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEMEMORY_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(
    nproc,
    [&pipes, &params](int rank, int world_size) {
      EXPECT_EQ(wholememory_init(0), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);

      wholememory_comm_t wm_comm = create_communicator_by_pipes(pipes, rank, world_size);

      if (wholememory_communicator_support_type_location(
            wm_comm, std::get<1>(params), std::get<2>(params)) != WHOLEMEMORY_SUCCESS) {
        EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);
        EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
        WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
        if (rank == 0) GTEST_SKIP_("Skip due to not supported.");
        return;
      }

      wholememory_handle_t handle1;
      EXPECT_EQ(wholememory::create_wholememory(&handle1,
                                                std::get<0>(params),
                                                wm_comm,
                                                std::get<1>(params),
                                                std::get<2>(params),
                                                std::get<2>(params)),
                WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory::destroy_wholememory(handle1), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);

      WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
    },
    true);
  ClosePipes(&pipes);
}

INSTANTIATE_TEST_SUITE_P(
  WholeMemoryHandleTests,
  WholeMemoryHandleCreateDestroyParameterTests,
  ::testing::Values(
    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_DEVICE, 128UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_HOST, 128UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CHUNKED, WHOLEMEMORY_ML_DEVICE, 128UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CHUNKED, WHOLEMEMORY_ML_HOST, 128UL),
    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_ML_DEVICE, 128UL),
    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_ML_HOST, 128UL),

    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_DEVICE, 63UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_HOST, 63UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CHUNKED, WHOLEMEMORY_ML_DEVICE, 63UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CHUNKED, WHOLEMEMORY_ML_HOST, 63UL),
    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_ML_DEVICE, 63UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_ML_HOST, 63UL),

    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_HOST, 128UL)));

class WholeMemoryHandleMultiCreateParameterTests
  : public ::testing::TestWithParam<
      std::tuple<wholememory_memory_type_t, wholememory_memory_location_t>> {};

TEST_P(WholeMemoryHandleMultiCreateParameterTests, CreateDestroyTest)
{
  auto params   = GetParam();
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEMEMORY_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(
    nproc,
    [&pipes, &params](int rank, int world_size) {
      EXPECT_EQ(wholememory_init(0), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);

      wholememory_comm_t wm_comm = create_communicator_by_pipes(pipes, rank, world_size);

      if (wholememory_communicator_support_type_location(
            wm_comm, std::get<0>(params), std::get<1>(params)) != WHOLEMEMORY_SUCCESS) {
        EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);
        EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
        WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
        if (rank == 0) GTEST_SKIP_("Skip due to not supported.");
        return;
      }

      size_t total_size  = 1024UL * 1024UL * 32;
      size_t granularity = 128;

      wholememory_handle_t handle1, handle2, handle3, handle4, handle5;
      EXPECT_EQ(
        wholememory::create_wholememory(
          &handle1, total_size, wm_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEMEMORY_SUCCESS);
      // handle1: 0
      EXPECT_EQ(handle1->handle_id, 0);

      EXPECT_EQ(
        wholememory::create_wholememory(
          &handle2, total_size, wm_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEMEMORY_SUCCESS);
      // handle1: 0, handle2: 1
      EXPECT_EQ(handle2->handle_id, 1);

      EXPECT_EQ(
        wholememory::create_wholememory(
          &handle3, total_size, wm_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEMEMORY_SUCCESS);
      // handle1: 0, handle2: 1, handle3: 2
      EXPECT_EQ(handle3->handle_id, 2);
      EXPECT_EQ(wm_comm->wholememory_map.size(), 3);

      EXPECT_EQ(wholememory::destroy_wholememory(handle2), WHOLEMEMORY_SUCCESS);
      // handle1: 0, handle3: 2
      EXPECT_EQ(wm_comm->wholememory_map.size(), 2);

      EXPECT_EQ(
        wholememory::create_wholememory(
          &handle4, total_size, wm_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEMEMORY_SUCCESS);
      // handle1: 0, handle4: 1, handle3: 2
      EXPECT_EQ(handle4->handle_id, 1);

      EXPECT_EQ(wholememory::destroy_wholememory(handle1), WHOLEMEMORY_SUCCESS);
      // handle4: 1, handle3: 2
      EXPECT_EQ(wm_comm->wholememory_map.size(), 2);

      EXPECT_EQ(wholememory::destroy_wholememory(handle3), WHOLEMEMORY_SUCCESS);
      // handle4: 1
      EXPECT_EQ(wm_comm->wholememory_map.size(), 1);

      EXPECT_EQ(
        wholememory::create_wholememory(
          &handle5, total_size, wm_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEMEMORY_SUCCESS);
      // handle5: 0, handle4: 1
      EXPECT_EQ(handle5->handle_id, 0);

      EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);

      WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
    },
    true);
  ClosePipes(&pipes);
}

#if 1
INSTANTIATE_TEST_SUITE_P(
  WholeMemoryHandleTests,
  WholeMemoryHandleMultiCreateParameterTests,
  ::testing::Values(std::make_tuple(WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_HOST),
                    std::make_tuple(WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_DEVICE),
                    std::make_tuple(WHOLEMEMORY_MT_CHUNKED, WHOLEMEMORY_ML_HOST),
                    std::make_tuple(WHOLEMEMORY_MT_CHUNKED, WHOLEMEMORY_ML_DEVICE),
                    std::make_tuple(WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_ML_HOST),
                    std::make_tuple(WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_ML_DEVICE)));
#endif
