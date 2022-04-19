#include "domain_socket_communicator.h"

#include <assert.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/un.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <vector>

namespace whole_memory {

class DomainSocketCommunicatorImpl {
 public:
  DomainSocketCommunicatorImpl() = default;
  ~DomainSocketCommunicatorImpl() {
    for (int i = 0; i < local_size_; i++) {
      close(recv_socks_[i]);
      assert(unlink(GetSocketName(i, local_rank_, local_size_).c_str()) == 0);
    }
    close(send_sock_);
    assert(unlink(GetSendSocketName(local_rank_, local_size_).c_str()) == 0);
  }
  void SetRankAndSize(int rank, int size) {
    local_rank_ = rank;
    local_size_ = size;
    recv_socks_.resize(size, -1);
    send_to_socket_names_.resize(size);
    for (int i = 0; i < size; i++) {
      recv_socks_[i] = OpenSocket(GetSocketName(i, rank, size));
      send_to_socket_names_[i] = GetSocketName(rank, i, size);
    }
    send_sock_ = OpenSocket(GetSendSocketName(rank, size));
    for (int i = 0; i < size; i++) {
      WaitSocketReady(GetSocketName(rank, i, size));
    }
  }
  int Rank() const {
    return local_rank_;
  }
  int Size() const {
    return local_size_;
  }
  void AllToAll(const void *send_buf,
                int send_size,
                void *recv_buf,
                int recv_size,
                const int *ranks,
                int rank_count) {
    int all_rank_count = (ranks == nullptr || rank_count == 0) ? local_size_ : rank_count;
    std::vector<int> ranks_vec(all_rank_count);
    for (int i = 0; i < all_rank_count; i++) {
      if (ranks == nullptr || rank_count == 0) {
        ranks_vec[i] = i;
      } else {
        ranks_vec[i] = ranks[i];
      }
    }
    for (int i = 0; i < all_rank_count; i++) {
      SendTo(send_sock_, send_to_socket_names_[ranks_vec[i]], (const char*)send_buf + i * send_size, send_size);
    }
    for (int i = 0; i < all_rank_count; i++) {
      RecvFrom(recv_socks_[ranks_vec[i]], (char*)recv_buf + i * recv_size, recv_size);
    }
  }
  void AllToAllV(const void ** send_bufs,
                 const int* send_sizes,
                 void ** recv_bufs,
                 const int* recv_sizes,
                 const int *ranks,
                 int rank_count) {
    int all_rank_count = (ranks == nullptr || rank_count == 0) ? local_size_ : rank_count;
    std::vector<int> ranks_vec(all_rank_count);
    for (int i = 0; i < all_rank_count; i++) {
      if (ranks == nullptr || rank_count == 0) {
        ranks_vec[i] = i;
      } else {
        ranks_vec[i] = ranks[i];
      }
    }
    for (int i = 0; i < all_rank_count; i++) {
      SendTo(send_sock_, send_to_socket_names_[ranks_vec[i]], (const char*)send_bufs[i], send_sizes[i]);
    }
    for (int i = 0; i < all_rank_count; i++) {
      RecvFrom(recv_socks_[ranks_vec[i]], (char*)recv_bufs[i], recv_sizes[i]);
    }
  }
 private:
  static std::string GetSocketName(int send_rank, int recv_rank, int size) {
    std::string filename = "/tmp/";
    const char* sock_prefix = getenv("WHOLEMEMORY_TMPNAME");
    std::string wholememory_prefix_str = "wmtmp";
    if (sock_prefix != nullptr) {
      wholememory_prefix_str = sock_prefix;
    }
    filename += wholememory_prefix_str;
    filename += "_sock_rank_";
    filename += std::to_string(send_rank);
    filename += "_sendto_";
    filename += std::to_string(recv_rank);
    filename += "_of_size_";
    filename += std::to_string(size);
    filename += ".sock";
    return filename;
  }
  static std::string GetSendSocketName(int send_rank, int size) {
    std::string filename = "/tmp/";
    const char* sock_prefix = getenv("WHOLEMEMORY_TMPNAME");
    std::string wholememory_prefix_str = "wmtmp";
    if (sock_prefix != nullptr) {
      wholememory_prefix_str = sock_prefix;
    }
    filename += wholememory_prefix_str;
    filename += "_sock_sendfrom_";
    filename += std::to_string(send_rank);
    filename += "_of_size_";
    filename += std::to_string(size);
    filename += ".sock";
    return filename;
  }
  static int OpenSocket(const std::string& socket_name) {
    int sock = 0;
    struct sockaddr_un cliaddr{};

    if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
      std::cerr << "DomainSocketCommunicatorImpl::OpenSocket failed: Socket creation error" << std::endl;
      abort();
      return -1;
    }
    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;

    strcpy(cliaddr.sun_path, socket_name.c_str());
    if (bind(sock, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
      std::cerr << "DomainSocketCommunicatorImpl::OpenSocket failed: Binding socket failed, name=" << socket_name << std::endl;
      abort();
      return -1;
    }
    return sock;
  }
  static void WaitSocketReady(const std::string& socket_name) {
    do {
      struct stat st_buffer;
      int retv = stat(socket_name.c_str(), &st_buffer);
      if (retv == 0) {
        if (S_ISSOCK(st_buffer.st_mode)) break;
        std::cerr << "[DomainSocketCommunicatorImpl::WaitSocketReady] socket " << socket_name
                  << " exist but not socket" << std::endl;
        abort();
      }
      usleep(10 * 1000);
    } while (true);
  }
  static void SendTo(int sendfd, const std::string& target, const void* data, size_t size) {
    struct sockaddr_un cliaddr{};
    // Construct client address to send this Shareable handle to
    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    strcpy(cliaddr.sun_path, target.c_str());
    auto send_success_size = sendto(sendfd, data, size, 0, (const sockaddr*)&cliaddr, sizeof(cliaddr));
    if (send_success_size != (ssize_t)size) {
      std::cerr << "[DomainSocketCommunicatorImpl::WaitSocketReady] send_size=" << size
                << ", returned=" << send_success_size << std::endl;
      abort();
    }
  }
  static void RecvFrom(int recvfd, void* data, size_t size) {
    assert(recv(recvfd, data, size, 0) == (ssize_t)size);
  }

  int local_size_ = 0;
  int local_rank_ = -1;
  int send_sock_ = -1;
  std::vector<int> recv_socks_;
  std::vector<std::string> send_to_socket_names_;
};

DomainSocketCommunicator::DomainSocketCommunicator() {
  impl_ = new DomainSocketCommunicatorImpl();
}

DomainSocketCommunicator::~DomainSocketCommunicator() {
  delete impl_;
  impl_ = nullptr;
}

void DomainSocketCommunicator::SetRankAndSize(int rank, int size) {
  impl_->SetRankAndSize(rank, size);
}

int DomainSocketCommunicator::Size() {
  return impl_->Size();
}

int DomainSocketCommunicator::Rank() {
  return impl_->Rank();
}

void DomainSocketCommunicator::AllToAll(const void *send_buf,
                               int send_size,
                               void *recv_buf,
                               int recv_size,
                               const int *ranks,
                               int rank_count) {
  impl_->AllToAll(send_buf, send_size, recv_buf, recv_size, ranks, rank_count);
}

void DomainSocketCommunicator::AllToAllV(const void **send_bufs,
                                         const int *send_sizes,
                                         void **recv_bufs,
                                         const int *recv_sizes,
                                         const int *ranks,
                                         int rank_count) {
  impl_->AllToAllV(send_bufs, send_sizes, recv_bufs, recv_sizes, ranks, rank_count);
}

}