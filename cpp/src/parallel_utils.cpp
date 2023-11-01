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
#include <string.h>
#include <unistd.h>
#include <wait.h>

#include <atomic>
#include <iostream>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "cuda_macros.hpp"
#include "net_utils.h"

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
  static bool is_child      = false;
  int child_idx             = 0;
  int current_running_count = running_count.fetch_add(1);
  if (current_running_count > 0 || is_child) {
    if (!is_child) running_count.fetch_sub(1);
    WHOLEMEMORY_FATAL("Already have MultiProcessRun, running_count=%d, %s child process",
                      current_running_count,
                      is_child ? "is" : "not");
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
      running_count.fetch_sub(1);
      WHOLEMEMORY_FATAL(
        "Rank %d returned pid %d not equal to pid %d", i, (int)pid_ret, (int)pids[i]);
    }
    if ((!WIFEXITED(wstatus)) || (WEXITSTATUS(wstatus) != 0)) {
      running_count.fetch_sub(1);
      WHOLEMEMORY_FATAL("Rank %d exit with error", i);
    }
  }
  running_count.fetch_sub(1);
}

class SideBandCommunicator {
 public:
  SideBandCommunicator(int world_rank, int world_size, const char* server_addr, int port);
  ~SideBandCommunicator();
  void Start();
  void Stop();
  void GroupAllToAll(const void* input, void* output, size_t element_size, int group_count = 1);
  void GroupAllGather(const void* input, void* output, size_t element_size, int group_count = 1);
  void GroupBroadcast(void* data, size_t element_size, int root_group_rank, int group_count = 1);
  void Barrier();

 private:
  static constexpr int kSideBandMagic = 0x51debacd;
  void ServerAcceptFunc();
  int world_rank_ = -1;
  int world_size_ = 0;
  std::string server_address_;
  int server_port_ = -1;
  int client_fd_   = -1;
  std::vector<int> server_fds_;
  std::thread server_thread_;
};

SideBandCommunicator::SideBandCommunicator(int world_rank,
                                           int world_size,
                                           const char* server_addr,
                                           int port)
  : server_address_(server_addr),
    server_port_(port),
    world_rank_(world_rank),
    world_size_(world_size)
{
}

SideBandCommunicator::~SideBandCommunicator() {}

void SideBandCommunicator::Start()
{
  server_fds_.resize(world_size_, -1);
  std::thread server_accept_thread;
  if (world_rank_ == 0) {
    server_accept_thread = std::thread([this]() { this->ServerAcceptFunc(); });
  }
  client_fd_ = CreateClientFd(server_address_, server_port_);
  int send_data[2];
  send_data[0] = kSideBandMagic;
  send_data[1] = world_rank_;
  SingleSend(client_fd_, &send_data[0], sizeof(int) * 2);
  int magic_number = 0;
  SingleRecv(client_fd_, &magic_number, sizeof(int));
  WHOLEMEMORY_CHECK_NOTHROW(magic_number == kSideBandMagic);
  if (world_rank_ == 0) { server_accept_thread.join(); }
  WHOLEMEMORY_INFO("[Client] Rank=%d connected to server.", world_rank_);
}

void SideBandCommunicator::Stop()
{
  WHOLEMEMORY_CHECK_NOTHROW(close(client_fd_) == 0);
  client_fd_ = -1;
  if (world_rank_ == 0) {
    for (int i = 0; i < world_size_; i++) {
      WHOLEMEMORY_CHECK_NOTHROW(close(server_fds_[i]) == 0);
      server_fds_[i] = -1;
    }
    server_fds_.clear();
  }
}

void SideBandCommunicator::ServerAcceptFunc()
{
  int server_listen_fd = CreateServerListenFd(server_port_);
  // Listening
  ServerListen(server_listen_fd, world_size_);

  std::set<int> unconnected_rank_set;
  for (int i = 0; i < world_size_; i++) {
    unconnected_rank_set.insert(i);
  }
  while (!unconnected_rank_set.empty()) {
    sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    int client_sock           = accept(server_listen_fd, (sockaddr*)&client_addr, &client_addr_len);
    if (client_sock >= 0) {
      int recv_data[2];
      SingleRecv(client_sock, &recv_data[0], sizeof(int) * 2);
      WHOLEMEMORY_CHECK_NOTHROW(recv_data[0] == kSideBandMagic);
      int rank_id = recv_data[1];
      WHOLEMEMORY_CHECK_NOTHROW(rank_id >= 0 && rank_id < world_size_);
      WHOLEMEMORY_CHECK_NOTHROW(unconnected_rank_set.count(rank_id) > 0);
      server_fds_[rank_id] = client_sock;
      unconnected_rank_set.erase(rank_id);
      WHOLEMEMORY_INFO("[Server] Rank %d connected to SideBandCommunicator", rank_id);
    }
  }
  WHOLEMEMORY_CHECK_NOTHROW(close(server_listen_fd) == 0);
  WHOLEMEMORY_INFO("[Server] All ranks connected to SideBandCommunicator");
  for (int i = 0; i < world_size_; i++) {
    int send_data[2];
    send_data[0] = kSideBandMagic;
    send_data[1] = i;
    SingleSend(server_fds_[i], &send_data[0], sizeof(int));
  }
}

void SideBandCommunicator::GroupAllToAll(const void* input,
                                         void* output,
                                         size_t element_size,
                                         int group_count)
{
  WHOLEMEMORY_CHECK_NOTHROW(world_size_ % group_count == 0);
  int group_size = world_size_ / group_count;
  SingleSend(client_fd_, input, element_size * group_size);
  if (world_rank_ == 0) {
    std::vector<char> recv_buffer(element_size * group_size);
    std::vector<std::vector<char>> send_buffer(group_size);
    for (int i = 0; i < group_size; i++) {
      send_buffer[i].resize(element_size * group_size);
    }
    for (int group_id = 0; group_id < group_count; group_id++) {
      for (int src_rank = group_id * group_size; src_rank < (group_id + 1) * group_size;
           src_rank++) {
        SingleRecv(server_fds_[src_rank], recv_buffer.data(), recv_buffer.size());
        for (int gr = 0; gr < group_size; gr++) {
          int src_gr = src_rank - group_id * group_size;
          memcpy(send_buffer[gr].data() + src_gr * element_size,
                 recv_buffer.data() + gr * element_size,
                 element_size);
        }
        for (int dst_gr = 0; dst_gr < group_size; dst_gr++) {
          int r = dst_gr + group_id * group_size;
          SingleSend(server_fds_[r], send_buffer[dst_gr].data(), send_buffer[dst_gr].size());
        }
      }
    }
  }
  SingleRecv(client_fd_, output, element_size * group_size);
}

void SideBandCommunicator::GroupAllGather(const void* input,
                                          void* output,
                                          size_t element_size,
                                          int group_count)
{
  WHOLEMEMORY_CHECK_NOTHROW(world_size_ % group_count == 0);
  int group_size = world_size_ / group_count;
  SingleSend(client_fd_, input, element_size);
  if (world_rank_ == 0) {
    std::vector<char> recv_buffer(element_size);
    std::vector<std::vector<char>> send_buffer(group_size);
    for (int i = 0; i < group_size; i++) {
      send_buffer[i].resize(element_size * group_size);
    }
    for (int group_id = 0; group_id < group_count; group_id++) {
      for (int src_rank = group_id * group_size; src_rank < (group_id + 1) * group_size;
           src_rank++) {
        SingleRecv(server_fds_[src_rank], recv_buffer.data(), recv_buffer.size());
        for (int gr = 0; gr < group_size; gr++) {
          int src_gr = src_rank - group_id * group_size;
          memcpy(send_buffer[gr].data() + src_gr * element_size, recv_buffer.data(), element_size);
        }
        for (int dst_gr = 0; dst_gr < group_size; dst_gr++) {
          int r = dst_gr + group_id * group_size;
          SingleSend(server_fds_[r], send_buffer[dst_gr].data(), send_buffer[dst_gr].size());
        }
      }
    }
  }
  SingleRecv(client_fd_, output, element_size * group_size);
}

void SideBandCommunicator::GroupBroadcast(void* data,
                                          size_t element_size,
                                          int root_group_rank,
                                          int group_count)
{
  WHOLEMEMORY_CHECK_NOTHROW(world_size_ % group_count == 0);
  int group_size = world_size_ / group_count;
  int group_rank = world_rank_ % group_size;
  if (group_rank == root_group_rank) { SingleSend(client_fd_, data, element_size); }
  if (world_rank_ == 0) {
    std::vector<char> recv_buffer(element_size);
    for (int group_id = 0; group_id < group_count; group_id++) {
      int src_rank = group_id * group_size + root_group_rank;
      SingleRecv(server_fds_[src_rank], recv_buffer.data(), recv_buffer.size());
      for (int r = group_id * group_size; r < (group_id + 1) * group_size; r++) {
        SingleSend(server_fds_[r], recv_buffer.data(), recv_buffer.size());
      }
    }
  }
  SingleRecv(client_fd_, data, element_size);
}

void SideBandCommunicator::Barrier()
{
  int data = 0;
  std::vector<int> recv_data(world_size_);
  GroupAllGather(&data, recv_data.data(), sizeof(int), 1);
}

SideBandCommunicator* StartSidebandCommunicator(int world_rank,
                                                int world_size,
                                                const char* server_addr,
                                                int port)
{
  auto* side_band_communicator =
    new SideBandCommunicator(world_rank, world_size, server_addr, port);
  side_band_communicator->Start();
  return side_band_communicator;
}

void SideBandAllToAll(SideBandCommunicator* side_band_communicator,
                      const void* input,
                      void* output,
                      size_t element_size)
{
  side_band_communicator->GroupAllToAll(input, output, element_size);
}

void SideBandAllGather(SideBandCommunicator* side_band_communicator,
                       const void* input,
                       void* output,
                       size_t element_size)
{
  side_band_communicator->GroupAllGather(input, output, element_size);
}

void SideBandBroadcast(SideBandCommunicator* side_band_communicator,
                       void* data,
                       size_t element_size,
                       int root_rank)
{
  side_band_communicator->GroupBroadcast(data, element_size, root_rank);
}

void ShutDownSidebandCommunicator(SideBandCommunicator* side_band_communicator)
{
  side_band_communicator->Stop();
  delete side_band_communicator;
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
