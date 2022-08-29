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
#pragma once

#include <stdio.h>
#include <unistd.h>

#include <functional>
#include <vector>

/*!
 * Run f with size threads
 * @param size : thread count
 * @param f : thread function
 */
void MultiThreadRun(int size, std::function<void(int, int)> f);

/*!
 * Run f with size processes
 * @param size : process count
 * @param f : process function
 */
void MultiProcessRun(int size, std::function<void(int, int)> f);

inline int CreatePipes(std::vector<std::array<int, 2>> *pipes, int nproc) {
  pipes->resize(nproc);
  for (int i = 0; i < nproc; i++) {
    if (pipe((*pipes)[i].data()) == -1) {
      fprintf(stderr, "Create pipe failed.\n");
      return -1;
    }
  }
  return 0;
}

template<typename T>
inline void PipeBroadcast(int rank, int size, int root, const std::vector<std::array<int, 2>> &pipes, T *data) {
  if (rank == root) {
    for (int i = 0; i < size; i++) {
      auto wret = write(pipes[i][1], data, sizeof(T));
      if (wret != sizeof(T)) {
        fprintf(stderr, "write to pipe failed.\n");
        abort();
      }
    }
  }
  auto rret = read(pipes[rank][0], data, sizeof(T));
  if (rret != sizeof(T)) {
    fprintf(stderr, "read from pipe failed.\n");
    abort();
  }
}

template<typename T>
inline void PipeGroupBroadcast(int rank,
                               int size,
                               int group_root,
                               int group_size,
                               const std::vector<std::array<int, 2>> &pipes,
                               T *data) {
  if (rank % group_size == group_root) {
    for (int i = rank - group_root; i < rank - group_root + group_size; i++) {
      auto wret = write(pipes[i][1], data, sizeof(T));
      if (wret != sizeof(T)) {
        fprintf(stderr, "write to pipe failed.\n");
        abort();
      }
    }
  }
  auto rret = read(pipes[rank][0], data, sizeof(T));
  if (rret != sizeof(T)) {
    fprintf(stderr, "read from pipe failed.\n");
    abort();
  }
}

int ForkGetDeviceCount();
