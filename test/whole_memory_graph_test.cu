#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <cuda_runtime_api.h>

#include "whole_memory.h"
#include "whole_memory_graph.h"
#include "parallel_utils.h"
#include "whole_memory_test_utils.cuh"
#include "whole_chunked_memory.cuh"

bool generate_graph = false;
bool load_graph = false;

std::string file_prefix;
int64_t node_count = 1024LL * 1024LL * 1024LL + 102031;
int avg_deg = 20;
int part_count = 63;

static struct option long_options[] = {
    {"prefix",     required_argument,       0,             0 },
    {"nodecount",  optional_argument,       0,  0 },
    {"avgdeg",     optional_argument,       0,     0 },
    {"partcount",  optional_argument,       0,  0 },
    {0,         0,                 0,  0 }
};

// g: generate graph
// l: load graph
static const char* shortopts = "gl";

void ParseOpts(int argc, char** argv) {
  int c;
  std::string name;
  char* endptr;
  while (true) {
    int option_index = 0;
    c = getopt_long(argc, argv, shortopts, long_options, &option_index);
    if (c == -1)
      break;
    switch (c) {
      case 0:
        name = long_options[option_index].name;
        if (name == "prefix") {
          file_prefix = optarg;
        } else if (name == "nodecount") {
          if (optarg == nullptr) {
            printf("--%s should have a value.\n", name.c_str());
            exit(-1);
          }
          node_count = strtoll(optarg, &endptr, 10);
          if (*endptr != '\0') {
            printf("nodecount: %s invalid.\n", optarg);
            exit(-1);
          }
        } else if (name == "avgdeg") {
          if (optarg == nullptr) {
            printf("--%s should have a value.\n", name.c_str());
            exit(-1);
          }
          avg_deg = strtol(optarg, &endptr, 10);
          if (*endptr != '\0') {
            printf("avgdeg: %s invalid.\n", optarg);
            exit(-1);
          }
        } else if (name == "partcount") {
          if (optarg == nullptr) {
            printf("--%s should have a value.\n", name.c_str());
            exit(-1);
          }
          part_count = strtol(optarg, &endptr, 10);
          if (*endptr != '\0') {
            printf("partcount: %s invalid.\n", optarg);
            exit(-1);
          }
        }
        break;
      case 'g':
        assert(!load_graph);
        generate_graph = true;
        break;
      case 'l':
        assert(!generate_graph);
        load_graph = true;
        break;
      default:
        printf("?? getopt returned character code 0%o ??\n", c);
        break;
    }
  }
  if (!load_graph && !generate_graph) {
    printf("either -l (load graph) or -g (generate graph) should be set.\n");
    exit(-1);
  }
  printf("%s graph %s %s\n", load_graph ? "Loading" : "Generating", load_graph ? "from" : "to", file_prefix.c_str());
  printf("nodecount=%ld, avgdeg=%d, partcount=%d\n", node_count, avg_deg, part_count);
  if (file_prefix.empty()) {
    printf("prefix should not be empty.\n");
    exit(-1);
  }
  if (node_count <= 0 || avg_deg <= 0 || avg_deg > node_count || part_count <= 0) {
    printf("invalid config\n");
    exit(-1);
  }
}

void GenerateFunction(int tid, int tcount) {
  int64_t nid = tid;
  const int kEdgeBufferCount = 1024 * 1024;
  auto* buf = (int64_t*)malloc(kEdgeBufferCount * sizeof(int64_t) * 2);
  int buf_used = 0;
  std::string filename = file_prefix;
  filename.append("_part_").append(std::to_string(tid)).append("_of_").append(std::to_string(tcount));
  FILE* fp = fopen(filename.c_str(), "wb");
  for (; nid < node_count; nid += tcount) {
    RandomNumGen rng(0, nid);
    int degree = rng.RandomMod(avg_deg * 2);
    for (int dst_idx = 0; dst_idx < degree; dst_idx++) {
      int64_t dst = rng.RandomMod64(node_count);
      buf[buf_used * 2] = nid;
      buf[buf_used * 2 + 1] = dst;
      buf_used += 1;
      if (buf_used == kEdgeBufferCount) {
        assert(fwrite(buf, sizeof(int64_t) * 2, buf_used, fp) == buf_used);
        buf_used = 0;
      }
    }
  }
  if (buf_used > 0) {
    assert(fwrite(buf, sizeof(int64_t) * 2, buf_used, fp) == buf_used);
  }
  fclose(fp);
  free(buf);
}

void GenerateGraph() {
  MultiThreadRun(part_count, GenerateFunction);
}

__global__ void CheckEdgeKernel(whole_memory::WholeChunkedMemoryHandle* csr_row_ptr_handle,
                                whole_memory::WholeChunkedMemoryHandle* csr_col_idx_handle,
                                int64_t total_node_count,
                                int64_t total_edge_count,
                                int avg_degree) {
  int64_t node_id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (node_id >= total_node_count) return;
  whole_memory::PtrGen<const whole_memory::WholeChunkedMemoryHandle, int64_t> csr_row_ptr_gen(csr_row_ptr_handle);
  whole_memory::PtrGen<const whole_memory::WholeChunkedMemoryHandle, int64_t> csr_col_idx_gen(csr_col_idx_handle);
  int64_t node_edge_start = *csr_row_ptr_gen.At(node_id);
  int64_t node_edge_end = *csr_row_ptr_gen.At(node_id + 1);
  if (node_edge_start > total_edge_count || node_edge_end > total_edge_count) {
    printf("node_id=%ld, total_node=%ld, start=%ld, end=%ld, total_edges=%ld\n",
        node_id, total_node_count, node_edge_start, node_edge_end, total_edge_count);
  }
  int real_edge_count = (int)(node_edge_end - node_edge_start);
  RandomNumGen rng(0, node_id);
  int degree = rng.RandomMod(avg_degree * 2);
  auto state = rng.SaveState();
  assert(degree >= real_edge_count);
  for (int real_edge_idx = 0; real_edge_idx < real_edge_count; real_edge_idx++) {
    rng.LoadState(state);
    int64_t dst_id = *csr_col_idx_gen.At(node_edge_start + real_edge_idx);
    int idx = 0;
    for (; idx < degree; idx++) {
      if (rng.RandomMod64(total_node_count) == dst_id) break;
    }
    assert(idx < degree);
  }
}

void LoadGraphFunction(int pid, int pcount) {
  whole_memory::WholeMemoryInit();
  int dev_count;
  assert(cudaGetDeviceCount(&dev_count) == cudaSuccess);
  assert(dev_count > 0);
  int dev_id = pid % dev_count;
  assert(cudaSetDevice(dev_id) == cudaSuccess);
  whole_memory::WmmpInit(pid, pcount, nullptr);
  void* local_edge_buffer;
  void* local_feature_buffer;
  int64_t edge_count = 0;
  int64_t src_node_count = -1;
  int64_t dst_node_count = -1;
  printf("=>Rank=%d start loading edge data...\n", pid);
  int64_t local_edge_count = whole_memory::WmmpTraverseLoadEdgeDataFromFileList(&local_edge_buffer,
                                                                                &local_feature_buffer,
                                                                                &edge_count,
                                                                                &src_node_count,
                                                                                &dst_node_count,
                                                                                file_prefix,
                                                                                whole_memory::WMT_Int64,
                                                                                0);

  int64_t loaded_node_count = std::max(src_node_count, dst_node_count);
  printf("=>Rank=%d, edge_count=%ld, local_edge_count=%ld, src_node_count=%ld, dst_node_count=%ld, loaded_node_count=%ld, node_count=%ld\n",
      pid, edge_count, local_edge_count, src_node_count, dst_node_count, loaded_node_count, node_count);
  assert(loaded_node_count <= node_count);
  whole_memory::WholeChunkedMemory_t csr_row_ptr, csr_col_idx;
  whole_memory::WcmmpMalloc(&csr_row_ptr, (node_count + 1) * sizeof(int64_t), sizeof(int64_t));
  whole_memory::WcmmpMalloc(&csr_col_idx, edge_count * sizeof(int64_t), sizeof(int64_t));

  int64_t final_edge_count;
  assert(cudaGetLastError() == cudaSuccess);
  whole_memory::WmmpLoadToChunkedCSRGraphFromEdgeBuffer(csr_row_ptr, csr_col_idx, local_edge_buffer, local_edge_count, node_count, edge_count, &final_edge_count, whole_memory::WMT_Int64, true, false, false);
  assert(cudaGetLastError() == cudaSuccess);
  printf("=>Rank=%d, final_edge_count=%ld\n", pid, final_edge_count);
  whole_memory::WmmpFreeEdgeData(local_edge_buffer, local_feature_buffer);

  whole_memory::WholeChunkedMemoryHandle* csr_row_ptr_handle = whole_memory::GetDeviceChunkedHandle(csr_row_ptr, dev_id);
  whole_memory::WholeChunkedMemoryHandle* csr_col_idx_handle = whole_memory::GetDeviceChunkedHandle(csr_col_idx, dev_id);

  assert(cudaGetLastError() == cudaSuccess);
  int thread_per_block = 256;
  int block_count = (int)((loaded_node_count + thread_per_block - 1) / thread_per_block);
  printf("=>Rank=%d, block_count=%d\n", pid, block_count);
  CheckEdgeKernel<<<block_count, thread_per_block>>>(csr_row_ptr_handle, csr_col_idx_handle, loaded_node_count, final_edge_count, avg_deg);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("=>Rank=%d, cudaGetLastError=%s\n", pid, cudaGetErrorString(err));
    abort();
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("=>Rank=%d, cudaDeviceSynchronize=%s\n", pid, cudaGetErrorString(err));
    abort();
  }

  whole_memory::WmmpBarrier();
  whole_memory::WcmmpFree(csr_row_ptr);
  whole_memory::WcmmpFree(csr_col_idx);
  whole_memory::WholeMemoryFinalize();
}

void LoadGraph() {
  MultiProcessRun(8, LoadGraphFunction);
}

int main(int argc, char** argv) {
  ParseOpts(argc, argv);
  if (generate_graph) {
    GenerateGraph();
  }
  if (load_graph) {
    LoadGraph();
  }
  return 0;
}