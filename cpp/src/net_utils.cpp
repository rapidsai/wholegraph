/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include "net_utils.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <string>

#include "cuda_macros.hpp"

static void ResolveHostName(sockaddr_in* saddr, const std::string& host_name, int port)
{
  addrinfo hints = {0, AF_INET, SOCK_STREAM, IPPROTO_TCP, 0, nullptr, nullptr, nullptr};
  addrinfo* res;
  char port_buf[16];
  snprintf(port_buf, 16, "%d", port);
  int ret = getaddrinfo(host_name.c_str(), port_buf, &hints, &res);
  if (ret != 0) {
    printf("Resolve IP for host %s failed.\n", host_name.c_str());
    abort();
  }
  *saddr = *(sockaddr_in*)(res->ai_addr);
}

int CreateServerListenFd(int port)
{
  int server_sock = socket(AF_INET, SOCK_STREAM, 0);
  WHOLEMEMORY_CHECK_NOTHROW(server_sock >= 0);
  int enable = 1;
  WHOLEMEMORY_CHECK_NOTHROW(
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) == 0);

  // Binding
  sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(sockaddr_in));
  server_addr.sin_family      = AF_INET;
  server_addr.sin_port        = htons(port);
  server_addr.sin_addr.s_addr = htonl(INADDR_ANY);

  WHOLEMEMORY_CHECK_NOTHROW(bind(server_sock, (sockaddr*)&server_addr, sizeof(server_addr)) == 0);

  return server_sock;
}

void ServerListen(int listen_fd, int backlog)
{
  WHOLEMEMORY_CHECK_NOTHROW(listen(listen_fd, backlog) == 0);
}

int ServerAccept(int listen_fd, sockaddr_in* client_addr, socklen_t* client_addr_len)
{
  int client_sock = accept(listen_fd, (sockaddr*)client_addr, client_addr_len);
  return client_sock;
}

int CreateClientFd(const std::string& server_name, int server_port)
{
  int client_sock = socket(AF_INET, SOCK_STREAM, 0);
  WHOLEMEMORY_CHECK_NOTHROW(client_sock >= 0);

  sockaddr_in server_addr;
  ResolveHostName(&server_addr, server_name, server_port);

  WHOLEMEMORY_CHECK_NOTHROW(server_addr.sin_family == AF_INET);
  WHOLEMEMORY_CHECK_NOTHROW(server_addr.sin_port == htons(server_port));
#if 0
  inet_pton(AF_INET, server_name.c_str(), &server_addr.sin_addr);
#endif

  while (connect(client_sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    switch (errno) {
      case ECONNREFUSED:
        // std::cerr << "Server may not running, waiting..." << std::endl;
        break;
      case ETIMEDOUT: printf("Connecting timeout retrying...\n"); break;
      case ENETUNREACH: printf("Network unreachable, retrying...\n"); break;
      default: printf("unknow error %d, retrying...\n", errno); break;
    }
    usleep(500 * 1000);
  }

  return client_sock;
}

void SingleSend(int sock_fd, const void* send_data, size_t send_size)
{
  ssize_t bytes_send = send(sock_fd, send_data, send_size, 0);
  if (bytes_send < 0) {
    printf("recv returned %ld, errno=%d %s\n", bytes_send, errno, strerror(errno));
  }
  WHOLEMEMORY_CHECK_NOTHROW(bytes_send == send_size);
}

void SingleRecv(int sock_fd, void* recv_data, size_t recv_size)
{
  ssize_t bytes_received = recv(sock_fd, recv_data, recv_size, 0);
  if (bytes_received < 0) {
    printf("recv returned %ld, errno=%d %s\n", bytes_received, errno, strerror(errno));
  }
  WHOLEMEMORY_CHECK_NOTHROW(bytes_received == recv_size);
}
