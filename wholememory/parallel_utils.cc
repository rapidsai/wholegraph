#include "parallel_utils.h"

#include <unistd.h>
#include <wait.h>

#include <iostream>
#include <vector>
#include <memory>
#include <thread>

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
