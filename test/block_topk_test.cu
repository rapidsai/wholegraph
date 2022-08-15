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
#include <assert.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include <algorithm>
#include <experimental/random>
#include <functional>
#include <random>
#include <set>

#include "../wholegraph/block_radix_topk.cuh"
#include "../wholegraph/macros.h"

const int test_case_k = 19;
const int test_case_m = 45;
const float test_cast[45] = {
    -0.844, 0.207, -0.258, 0.484, 0.461, -0.543, -0.064, -0.885, 0.981, -0.828,
    -0.207, 0.980, -0.761, 0.098, 0.849, 0.113, 0.176, 0.939, 0.212, -0.669,
    0.943, 0.009, 0.113, 0.657, -0.608, 0.115, -0.677, 0.077, -0.818, 0.423,
    0.440, -0.958, 0.350, 0.038, -0.528, -0.622, -0.989, -0.260, 0.245, 0.722,
    -0.446, -0.648, 0.756, -0.847, -0.376};

void FixedDataGenerator(float *data, int size) {
  if (size != test_case_m) {
    fprintf(stderr, "This FixedDataGenerator is only for size=%d\n", test_case_m);
    abort();
  }
  for (int i = 0; i < test_case_m; i++) {
    data[i] = test_cast[i];
  }
}

void RandomUniformFloatArray(float *data, int size, float min_value, float max_value) {
  static std::random_device rd;
  static std::mt19937 e2(rd());
  std::uniform_real_distribution<float> dist(min_value, max_value);
  for (int i = 0; i < size; i++) {
    data[i] = dist(e2);
  }
}
void RandomUniformIntFloatArray(float *data, int size, int min_value, int max_value) {
  std::uniform_real_distribution<float> dist(min_value, max_value);
  for (int i = 0; i < size; i++) {
    data[i] = (float) std::experimental::randint(min_value, max_value);
  }
}

struct CPUReference {
  std::set<int> larger_idx_set;
  std::set<int> equal_idx_set;
  std::vector<float> sorted_data;
  float threshold;
};

void CPUTopKReference(const float *data, int k, int size, CPUReference *ref) {
  ref->sorted_data.resize(size);
  std::copy(data, data + size, ref->sorted_data.begin());
  std::sort(ref->sorted_data.begin(), ref->sorted_data.end(), std::greater<float>());
  float threshold = ref->sorted_data[k - 1];
  ref->threshold = threshold;
  for (int i = 0; i < size; i++) {
    if (data[i] > threshold) {
      ref->larger_idx_set.insert(i);
    } else if (data[i] == threshold) {
      ref->equal_idx_set.insert(i);
    }
  }
}

template<int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void TopKRegisterKernel(const float *data, int k, int size, float *topk_weight, int *topk_indice) {
  using BlockRadixTopK = whole_graph::BlockRadixTopKRegister<float, BLOCK_SIZE, ITEMS_PER_THREAD, true, int>;
  __shared__ typename BlockRadixTopK::TempStorage temp_storage;
  assert(size <= BLOCK_SIZE * ITEMS_PER_THREAD);
  float keys[ITEMS_PER_THREAD];
  int indices[ITEMS_PER_THREAD];
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx = i * BLOCK_SIZE + threadIdx.x;
    if (idx < size) keys[i] = data[idx];
    indices[i] = idx;
  }
  BlockRadixTopK{temp_storage}.radixTopKToStriped(keys, indices, k, size);
  __syncthreads();
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx = i * BLOCK_SIZE + threadIdx.x;
    if (idx < k) {
      topk_weight[idx] = keys[i];
      topk_indice[idx] = indices[i];
    }
  }
}

bool CompareResults(const float *gpu_weights, const int *gpu_indice, const float *weights, CPUReference *ref, int k, int m) {
  float threshold = ref->threshold;
  std::set<int> larger_set, equal_set;
  for (int i = 0; i < k; i++) {
    float w = gpu_weights[i];
    int idx = gpu_indice[i];
    if (weights[idx] != w) {
      fprintf(stderr, "gpu_weights[%d]=%f with gpu_indice[%d]=%d, but raw weights[%d]=%f\n", i, w, i, idx, idx, weights[idx]);
      return false;
    }
    if (larger_set.find(idx) != larger_set.end() || equal_set.find(idx) != equal_set.end()) {
      fprintf(stderr, "gpu_weights[%d]=%f with gpu_indice[%d]=%d, idx already found before.\n", i, w, i, idx);
      return false;
    }
    if (w > threshold) {
      larger_set.insert(idx);
      if (ref->larger_idx_set.find(idx) == ref->larger_idx_set.end()) {
        fprintf(stderr, "index=%d, weight=%f found by gpu, but not in CPU ref larger_set, threshold=%f\n", idx, w, threshold);
        return false;
      }
    } else if (w == threshold) {
      equal_set.insert(idx);
      if (ref->equal_idx_set.find(idx) == ref->equal_idx_set.end()) {
        fprintf(stderr, "index=%d, weight=%f found by gpu, but not in CPU ref equal_set, threshold=%f\n", idx, w, threshold);
        return false;
      }
    } else {
      fprintf(stderr, "index=%d, weight=%f, less than threshold=%f\n", idx, w, threshold);
      return false;
    }
  }
  return true;
}

void PrintWeights(const char *prefix, const float *weights, int size) {
  fprintf(stderr, "%s len=%d:", prefix, size);
  const int item_per_line = 10;
  for (int i = 0; i < size; i++) {
    if (i % item_per_line == 0) {
      fprintf(stderr, "\n[%4d-%4d]: ", i, i + item_per_line - 1);
    }
    fprintf(stderr, "%8.4f,", weights[i]);
  }
  fprintf(stderr, "\n");
}

void PrintIndice(const char *prefix, const int *indice, int size) {
  fprintf(stderr, "%s len=%d:", prefix, size);
  const int item_per_line = 10;
  for (int i = 0; i < size; i++) {
    if (i % item_per_line == 0) {
      fprintf(stderr, "\n[%4d-%4d]: ", i, i + item_per_line - 1);
    }
    fprintf(stderr, "%8d", indice[i]);
  }
  fprintf(stderr, "\n");
}

template<int BLOCK_SIZE, int ITEMS_PER_THREAD>
void BlockRadixTopKRegisterTestCase(int k, int m, std::function<void(float *, int)> generator) {
  float *weights = nullptr;
  float *gpu_weights = nullptr;
  int *gpu_indice = nullptr;
  bool ret = true;
  WM_CUDA_CHECK(cudaMallocManaged(&weights, sizeof(float) * m));
  WM_CUDA_CHECK(cudaMallocManaged(&gpu_weights, sizeof(float) * k));
  WM_CUDA_CHECK(cudaMallocManaged(&gpu_indice, sizeof(int) * k));

  generator(weights, m);
  CPUReference cpu_ref;
  CPUTopKReference(weights, k, m, &cpu_ref);

  TopKRegisterKernel<BLOCK_SIZE, ITEMS_PER_THREAD><<<1, BLOCK_SIZE>>>(weights, k, m, gpu_weights, gpu_indice);
  WM_CUDA_CHECK(cudaGetLastError());
  cudaError_t err = cudaDeviceSynchronize();

  if (err == cudaSuccess) {
    ret = CompareResults(gpu_weights, gpu_indice, weights, &cpu_ref, k, m);
  }

  if (!ret || err != cudaSuccess) {
    fprintf(stderr, "[Register] BLOCK_SIZE=%d, ITEMS_PER_THREAD=%d, k=%d, m=%d\n", BLOCK_SIZE, ITEMS_PER_THREAD, k, m);
    if (err != cudaSuccess) fprintf(stderr, "Kernel failed.\n");
    PrintWeights("[Register] Raw weights", weights, m);
    PrintWeights("[Register] CPU Ref sorted", cpu_ref.sorted_data.data(), m);
    fprintf(stderr, "[Register] CPU Ref threshold=%f\n", cpu_ref.threshold);
    if (err == cudaSuccess) {
      PrintWeights("[Register] GPU topK weights", gpu_weights, k);
      PrintIndice("[Register] GPU topK indice", gpu_indice, k);
    }
    abort();
  }

  WM_CUDA_CHECK(cudaFree(weights));
  WM_CUDA_CHECK(cudaFree(gpu_weights));
  WM_CUDA_CHECK(cudaFree(gpu_indice));
}

void RepeatTest(std::function<void()> f, int count) {
  for (int t = 0; t < count; t++) {
    f();
  }
}

template<int BLOCK_SIZE, int ITEMS_PER_THREAD>
void BlockRadixTopKRegisterTestWithRepeat(int repeat_count) {
  RepeatTest([]() {
    std::random_device rd;
    std::mt19937 e2(rd());
    int k = std::experimental::randint<int>(1, BLOCK_SIZE * ITEMS_PER_THREAD - 1);
    int m = std::experimental::randint<int>(k + 1, BLOCK_SIZE * ITEMS_PER_THREAD);
    fprintf(stderr, "[Register] Testing with m=%d, k=%d\n", m, k);
    BlockRadixTopKRegisterTestCase<BLOCK_SIZE, ITEMS_PER_THREAD>(k, m, std::bind(RandomUniformFloatArray, std::placeholders::_1, std::placeholders::_2, -1., 1.));
    BlockRadixTopKRegisterTestCase<BLOCK_SIZE, ITEMS_PER_THREAD>(k, m, std::bind(RandomUniformIntFloatArray, std::placeholders::_1, std::placeholders::_2, -1000, 1000));
  },
             repeat_count);
}

void BlockRadixTopKRegisterTest() {
  BlockRadixTopKRegisterTestCase<64, 1>(test_case_k, test_case_m, FixedDataGenerator);
#if 1
  BlockRadixTopKRegisterTestWithRepeat<32, 1>(10000);
  BlockRadixTopKRegisterTestWithRepeat<64, 1>(10000);
  BlockRadixTopKRegisterTestWithRepeat<128, 1>(10000);
  BlockRadixTopKRegisterTestWithRepeat<256, 1>(10000);
  BlockRadixTopKRegisterTestWithRepeat<512, 1>(10000);
  BlockRadixTopKRegisterTestWithRepeat<32, 2>(10000);
  BlockRadixTopKRegisterTestWithRepeat<64, 2>(10000);
  BlockRadixTopKRegisterTestWithRepeat<128, 2>(10000);
  BlockRadixTopKRegisterTestWithRepeat<256, 2>(10000);
  BlockRadixTopKRegisterTestWithRepeat<512, 2>(10000);
  BlockRadixTopKRegisterTestWithRepeat<32, 4>(10000);
  BlockRadixTopKRegisterTestWithRepeat<64, 4>(10000);
  BlockRadixTopKRegisterTestWithRepeat<128, 4>(10000);
  BlockRadixTopKRegisterTestWithRepeat<256, 4>(10000);
  BlockRadixTopKRegisterTestWithRepeat<512, 4>(10000);
  BlockRadixTopKRegisterTestWithRepeat<32, 8>(10000);
  BlockRadixTopKRegisterTestWithRepeat<64, 8>(10000);
  BlockRadixTopKRegisterTestWithRepeat<128, 8>(10000);
  BlockRadixTopKRegisterTestWithRepeat<256, 8>(10000);
  BlockRadixTopKRegisterTestWithRepeat<512, 8>(10000);
#endif
  fprintf(stderr, "[Register] BlockRadixTopKRegisterTest all correct!\n");
}

template<int BLOCK_SIZE>
__global__ void TopKGlobalMemoryKernel(const float *data, int k, int size, float *topk_weight, int *topk_indice) {
  using BlockRadixTopK = whole_graph::BlockRadixTopKGlobalMemory<float, BLOCK_SIZE, true>;
  __shared__ typename BlockRadixTopK::TempStorage temp_storage;
  float threshold;
  bool is_unique;
  BlockRadixTopK{temp_storage}.radixTopKGetThreshold(data, k, size, threshold, is_unique);
  __shared__ int cnt;
  if (threadIdx.x == 0) cnt = 0;
  __syncthreads();
  //if (threadIdx.x == 0) printf("threshold=%f (%x), is_unique=%d\n", threshold, reinterpret_cast<int&>(threshold), is_unique ? 1 : 0);
  if (is_unique) {
    for (int i = threadIdx.x; i < size; i += BLOCK_SIZE) {
      float key = data[i];
      bool has_topk = (key >= threshold);
      if (has_topk) {
        int write_index = atomicAdd(&cnt, 1);
        topk_weight[write_index] = key;
        topk_indice[write_index] = i;
      }
    }
  } else {
    for (int i = threadIdx.x; i < size; i += BLOCK_SIZE) {
      float key = data[i];
      bool has_topk = (key > threshold);
      if (has_topk) {
        int write_index = atomicAdd(&cnt, 1);
        topk_weight[write_index] = key;
        topk_indice[write_index] = i;
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < size; i += BLOCK_SIZE) {
      float key = data[i];
      bool has_topk = (key == threshold);
      if (has_topk) {
        int write_index = atomicAdd(&cnt, 1);
        if (write_index >= k) break;
        topk_weight[write_index] = key;
        topk_indice[write_index] = i;
      }
    }
  }
}

template<int BLOCK_SIZE>
void BlockRadixTopKGlobalMemoryTestCase(int k, int m, std::function<void(float *, int)> generator) {
  float *weights = nullptr;
  float *gpu_weights = nullptr;
  int *gpu_indice = nullptr;
  bool ret = true;
  WM_CUDA_CHECK(cudaMallocManaged(&weights, sizeof(float) * m));
  WM_CUDA_CHECK(cudaMallocManaged(&gpu_weights, sizeof(float) * k));
  WM_CUDA_CHECK(cudaMallocManaged(&gpu_indice, sizeof(int) * k));

  generator(weights, m);
  CPUReference cpu_ref;
  CPUTopKReference(weights, k, m, &cpu_ref);

  TopKGlobalMemoryKernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(weights, k, m, gpu_weights, gpu_indice);
  WM_CUDA_CHECK(cudaGetLastError());
  cudaError_t err = cudaDeviceSynchronize();

  if (err == cudaSuccess) {
    ret = CompareResults(gpu_weights, gpu_indice, weights, &cpu_ref, k, m);
  }

  if (!ret || err != cudaSuccess) {
    fprintf(stderr, "[GlobalMemory] BLOCK_SIZE=%d, k=%d, m=%d\n", BLOCK_SIZE, k, m);
    if (err != cudaSuccess) fprintf(stderr, "[GlobalMemory] Kernel failed.\n");
    PrintWeights("[GlobalMemory] Raw weights", weights, m);
    PrintWeights("[GlobalMemory] CPU Ref sorted", cpu_ref.sorted_data.data(), m);
    fprintf(stderr, "[GlobalMemory] CPU Ref threshold=%f (%x)\n", cpu_ref.threshold, reinterpret_cast<int &>(cpu_ref.threshold));
    if (err == cudaSuccess) {
      PrintWeights("[GlobalMemory] GPU topK weights", gpu_weights, k);
      PrintIndice("[GlobalMemory] GPU topK indice", gpu_indice, k);
    }
    abort();
  }

  WM_CUDA_CHECK(cudaFree(weights));
  WM_CUDA_CHECK(cudaFree(gpu_weights));
  WM_CUDA_CHECK(cudaFree(gpu_indice));
}

template<int BLOCK_SIZE>
void BlockRadixTopKGlobalMemoryTestWithRepeat(int repeat_count) {
  RepeatTest([]() {
    std::random_device rd;
    std::mt19937 e2(rd());
    const int kMaxM = 10000;
    int k = std::experimental::randint<int>(1, kMaxM - 1);
    int m = std::experimental::randint<int>(k + 1, kMaxM);
    fprintf(stderr, "[GlobalMemory] Testing with m=%d, k=%d\n", m, k);
    BlockRadixTopKGlobalMemoryTestCase<BLOCK_SIZE>(k, m, std::bind(RandomUniformFloatArray, std::placeholders::_1, std::placeholders::_2, -1., 1.));
    BlockRadixTopKGlobalMemoryTestCase<BLOCK_SIZE>(k, m, std::bind(RandomUniformIntFloatArray, std::placeholders::_1, std::placeholders::_2, -1000, 1000));
  },
             repeat_count);
}

void BlockRadixTopKGlobalMemoryTest() {
  BlockRadixTopKGlobalMemoryTestCase<32>(10, 20, std::bind(RandomUniformIntFloatArray, std::placeholders::_1, std::placeholders::_2, -1000, -1));
#if 1
  BlockRadixTopKGlobalMemoryTestWithRepeat<32>(10000);
  BlockRadixTopKGlobalMemoryTestWithRepeat<64>(10000);
  BlockRadixTopKGlobalMemoryTestWithRepeat<128>(10000);
  BlockRadixTopKGlobalMemoryTestWithRepeat<256>(10000);
  BlockRadixTopKGlobalMemoryTestWithRepeat<512>(10000);
#endif
  fprintf(stderr, "[GlobalMemory] BlockRadixTopKGlobalMemoryTest all correct!\n");
}

int main(int argc, char **argv) {
  BlockRadixTopKRegisterTest();
  BlockRadixTopKGlobalMemoryTest();
  return 0;
}