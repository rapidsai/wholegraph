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