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
#pragma once

#include <netinet/in.h>
#include <sys/socket.h>

#include <string>

int CreateServerListenFd(int port);

void ServerListen(int listen_fd, int backlog = 10);

int ServerAccept(int listen_fd, sockaddr_in* client_addr, socklen_t* client_addr_len);

int CreateClientFd(const std::string& server_name, int server_port);

void SingleSend(int sock_fd, const void* send_data, size_t send_size);

void SingleRecv(int sock_fd, void* recv_data, size_t recv_size);
