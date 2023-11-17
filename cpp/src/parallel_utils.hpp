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
#pragma once

#include <unistd.h>

#include <functional>
#include <vector>

#include "logger.hpp"

/**
 * Run f with size threads
 * @param size : thread count
 * @param f : thread function
 */
void MultiThreadRun(int size, std::function<void(int, int)> f);

/**
 * Get processor count of the machine.
 * @return : processor count
 */
int GetProcessorCount();

/**
 * Run f with size processes
 * @note when using gtest with MultiProcessRun, testing::Test::HasFailure()
 * need to be called before f return and modify exit code according to if has
 * gtest failures. See parallel_utils_tests.cpp for reference.
 * @param size : process count
 * @param f : process function
 * @param inline_single_process : use current process to run f if size==1
 */
void MultiProcessRun(int size, std::function<void(int, int)> f, bool inline_single_process = false);

inline int CreatePipes(std::vector<std::array<int, 2>>* pipes, int nproc)
{
  pipes->resize(nproc);
  for (int i = 0; i < nproc; i++) {
    if (pipe((*pipes)[i].data()) == -1) {
      WHOLEMEMORY_ERROR("Create pipe failed.");
      return -1;
    }
  }
  return 0;
}

inline void ClosePipes(std::vector<std::array<int, 2>>* pipes)
{
  for (size_t i = 0; i < pipes->size(); i++) {
    WHOLEMEMORY_CHECK(close(pipes->at(i)[0]) == 0);
    WHOLEMEMORY_CHECK(close(pipes->at(i)[1]) == 0);
  }
  pipes->clear();
}

template <typename T>
inline void PipeBroadcast(
  int rank, int world_size, int root, const std::vector<std::array<int, 2>>& pipes, T* data)
{
  if (rank == root) {
    for (int i = 0; i < world_size; i++) {
      auto wret = write(pipes[i][1], data, sizeof(T));
      if (wret != sizeof(T)) { WHOLEMEMORY_FATAL("write to pipe failed."); }
    }
  }
  auto rret = read(pipes[rank][0], data, sizeof(T));
  if (rret != sizeof(T)) { WHOLEMEMORY_FATAL("read to pipe failed."); }
}

template <typename T>
inline void PipeGroupBroadcast(int rank,
                               int world_size,
                               int group_root,
                               int group_size,
                               const std::vector<std::array<int, 2>>& pipes,
                               T* data)
{
  WHOLEMEMORY_CHECK(world_size % group_size == 0);
  if (rank % group_size == group_root) {
    for (int i = rank - group_root; i < rank - group_root + group_size; i++) {
      auto wret = write(pipes[i][1], data, sizeof(T));
      if (wret != sizeof(T)) { WHOLEMEMORY_FATAL("write to pipe failed."); }
    }
  }
  auto rret = read(pipes[rank][0], data, sizeof(T));
  if (rret != sizeof(T)) { WHOLEMEMORY_FATAL("read to pipe failed."); }
}

class SideBandCommunicator;

SideBandCommunicator* StartSidebandCommunicator(int world_rank,
                                                int world_size,
                                                const char* server_addr,
                                                int port);
void SideBandAllToAll(SideBandCommunicator* side_band_communicator,
                      const void* input,
                      void* output,
                      size_t element_size);
void SideBandAllGather(SideBandCommunicator* side_band_communicator,
                       const void* input,
                       void* output,
                       size_t element_size);
void SideBandBroadcast(SideBandCommunicator* side_band_communicator,
                       void* data,
                       size_t element_size,
                       int root_rank);
void ShutDownSidebandCommunicator(SideBandCommunicator* side_band_communicator);

int ForkGetDeviceCount();
