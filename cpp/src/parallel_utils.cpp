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
#include "parallel_utils.hpp"

#include <cuda_runtime_api.h>
#include <unistd.h>
#include <wait.h>

#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "cuda_macros.hpp"

void MultiThreadRun(int size, std::function<void(int, int)> f)
{
  std::vector<std::unique_ptr<std::thread>> threads(size);
  for (int i = 0; i < size; i++) {
    threads[i] = std::make_unique<std::thread>([f, i, size] { return f(i, size); });
  }
  for (int i = 0; i < size; i++) {
    threads[i]->join();
  }
}

int GetProcessorCount() { return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN)); }

void MultiProcessRun(int world_size, std::function<void(int, int)> f, bool inline_single_process)
{
  if (world_size == 1 && inline_single_process) {
    f(0, 1);
    return;
  }
  // This variable is added to prevent from calling MultiProcessRun recursively by mistake,
  // which may fork too many process and lead to system crash.
  static std::atomic<int64_t> running_count(0);
  std::vector<pid_t> pids(world_size);
  bool is_child             = false;
  int child_idx             = 0;
  int current_running_count = running_count.fetch_add(1);
  if (current_running_count > 0) {
    WHOLEMEMORY_FATAL("Already have MultiProcessRun, running_count=%d", current_running_count);
  }
  for (; child_idx < world_size; child_idx++) {
    pids[child_idx] = fork();
    if (pids[child_idx] == -1) {
      WHOLEMEMORY_ERROR("fork failed.");
      break;
    }
    if (pids[child_idx] == 0) {
      is_child = true;
      f(child_idx, world_size);
      exit(0);
    }
  }
  if (child_idx != world_size) {
    for (int i = 0; i < child_idx; i++) {
      // kill all launched child process in case they may wait for each other.
      kill(pids[i], SIGKILL);
      int wstatus;
      pid_t pid_ret = waitpid(pids[i], &wstatus, 0);
    }
    WHOLEMEMORY_FATAL("MultiProcessRun failed.");
  }
  for (int i = 0; i < world_size; i++) {
    int wstatus;
    pid_t pid_ret = waitpid(pids[i], &wstatus, 0);
    if (pid_ret != pids[i]) {
      WHOLEMEMORY_FATAL(
        "Rank %d returned pid %d not equal to pid %d", i, (int)pid_ret, (int)pids[i]);
    }
    if ((!WIFEXITED(wstatus)) || (WEXITSTATUS(wstatus) != 0)) {
      WHOLEMEMORY_FATAL("Rank %d exit with error", i);
    }
  }
  running_count.fetch_sub(1);
}

int ForkGetDeviceCount()
{
  static int s_device_count = -1;
  if (s_device_count >= 0) { return s_device_count; }
  int pipes[2];
  if (pipe(pipes) == -1) {
    WHOLEMEMORY_ERROR("Create pipe failed.");
    return -1;
  }
  pid_t pid = fork();
  if (pid == -1) {
    WHOLEMEMORY_ERROR("fork failed.");
    return -1;
  }
  if (pid == 0) {
    int dev_count = -1;
    WM_CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    WHOLEMEMORY_CHECK(close(pipes[0]) == 0);
    auto wret = write(pipes[1], &dev_count, sizeof(int));
    if (wret != sizeof(int)) { WHOLEMEMORY_FATAL("write dev_count to pipe failed."); }
    WHOLEMEMORY_CHECK(close(pipes[1]) == 0);
    exit(0);
  } else {
    int dev_count = -1;
    WHOLEMEMORY_CHECK(close(pipes[1]) == 0);
    auto rret = read(pipes[0], &dev_count, sizeof(int));
    if (rret != sizeof(int)) { WHOLEMEMORY_FATAL("read dev_count from pipe failed."); }
    WHOLEMEMORY_CHECK(close(pipes[0]) == 0);
    int wstatus;
    pid_t pid_ret = waitpid(pid, &wstatus, 0);
    if (pid_ret != pid) { WHOLEMEMORY_FATAL("wait dev_count process failed."); }
    s_device_count = dev_count;
    return dev_count;
  }
}
