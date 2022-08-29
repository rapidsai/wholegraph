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
#include "parallel_utils.h"

#include <cuda_runtime_api.h>
#include <unistd.h>
#include <wait.h>

#include <iostream>
#include <memory>
#include <thread>
#include <vector>

void MultiThreadRun(int size, std::function<void(int, int)> f) {
  std::vector<std::unique_ptr<std::thread>> threads(size);
  for (int i = 0; i < size; i++) {
    threads[i] = std::make_unique<std::thread>([f, i, size] { return f(i, size); });
  }
  for (int i = 0; i < size; i++) {
    threads[i]->join();
  }
}

void MultiProcessRun(int size, std::function<void(int, int)> f) {
  std::vector<pid_t> pids(size);
  for (int i = 0; i < size; i++) {
    pids[i] = fork();
    if (pids[i] == -1) {
      std::cerr << "fork failed.\n";
      abort();
    }
    if (pids[i] == 0) {
      f(i, size);
      return;
    }
  }
  for (int i = 0; i < size; i++) {
    int wstatus;
    pid_t pid_ret = waitpid(pids[i], &wstatus, 0);
    if (pid_ret != pids[i]) {
      std::cerr << "Rank " << i << " return pid " << pid_ret << " not equales to pid " << pids[i] << ".\n";
    }
    if (!WIFEXITED(wstatus)) {
      std::cerr << "Rank " << i << " exit with error.\n";
    }
  }
}

int ForkGetDeviceCount() {
  int pipes[2];
  if (pipe(pipes) == -1) {
    fprintf(stderr, "Create pipe failed.\n");
    return -1;
  }
  pid_t pid = fork();
  if (pid == -1) {
    fprintf(stderr, "fork failed.\n");
    return -1;
  }
  if (pid == 0) {
    int dev_count = -1;
    auto err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess) {
      dev_count = -1;
    }
    auto wret = write(pipes[1], &dev_count, sizeof(int));
    if (wret != sizeof(int)) {
      fprintf(stderr, "write dev_count to pipe failed.\n");
      abort();
    }
    exit(0);
  } else {
    int dev_count = -1;
    auto rret = read(pipes[0], &dev_count, sizeof(int));
    if (rret != sizeof(int)) {
      fprintf(stderr, "read dev_count from pipe failed.\n");
      abort();
    }
    int wstatus;
    pid_t pid_ret = waitpid(pid, &wstatus, 0);
    if (pid_ret != pid) {
      fprintf(stderr, "wait dev_count process failed.\n");
      abort();
    }
    return dev_count;
  }
}