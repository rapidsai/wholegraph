/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>

#include <algorithm>
#include <experimental/random>
#include <iostream>
#include <random>

#include "wholememory_ops/functions/embedding_cache_func.cuh"

template <typename DataT>
static void PrintVector(const std::vector<DataT>& v,
                        const std::string& name,
                        int padding        = 0,
                        int value_per_line = 8)
{
  std::cout << std::string(padding, ' ') << "vector " << name << " length=" << v.size() << " :";
  for (size_t i = 0; i < v.size(); i++) {
    if (i % value_per_line == 0) std::cout << std::endl << std::string(padding + 2, ' ');
    std::cout << v[i] << ",\t";
  }
  std::cout << std::endl;
}
static void PrintTagVector(const std::vector<uint16_t>& v,
                           const std::string& name,
                           int padding        = 0,
                           int value_per_line = 8)
{
  std::cout << std::string(padding, ' ') << "vector " << name << " length=" << v.size() << " :";
  for (size_t i = 0; i < v.size(); i++) {
    if (i % value_per_line == 0) std::cout << std::endl << std::string(padding + 2, ' ');
    uint16_t tag  = v[i];
    bool valid    = tag & (1U << 14);
    bool modified = tag & (1U << 15);
    int lid       = valid ? (int)(tag & ((1U << 14U) - 1)) : -1;
    std::ostringstream oss;
    oss << (valid ? "[V" : "[I");
    oss << (modified ? "M" : " ");
    oss << std::setw(6) << std::setfill(' ') << lid;
    oss << "]";
    std::cout << oss.str() << "\t";
  }
  std::cout << std::endl;
}

static void PrintLfuCountVector(const std::vector<uint16_t>& v,
                                const std::string& name,
                                int padding        = 0,
                                int value_per_line = 8)
{
  std::cout << std::string(padding, ' ') << "vector " << name << " length=" << v.size() << " :";
  int scale = 0;
  for (size_t i = 0; i < v.size(); i++) {
    if (i % value_per_line == 0) std::cout << std::endl << std::string(padding + 2, ' ');
    int lfu_count = v[i] & ((1U << 14) - 1);
    std::cout << lfu_count << "\t";
    if (v[i] & (1U << 14)) { scale |= (1U << i); }
  }
  std::cout << std::endl;
  std::cout << std::string(padding + 2, ' ') << "lfu scale = " << scale << std::endl;
}

struct SingleCacheSetTestParam {
  std::vector<int> cache_tag_lids      = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  std::vector<bool> cache_tag_valid    = {false, false, false, false, false, false, false, false,
                                          false, false, false, false, false, false, false, false,
                                          false, false, false, false, false, false, false, false,
                                          false, false, false, false, false, false, false, false};
  std::vector<bool> cache_tag_modified = {false, false, false, false, false, false, false, false,
                                          false, false, false, false, false, false, false, false,
                                          false, false, false, false, false, false, false, false,
                                          false, false, false, false, false, false, false, false};
  ;
  std::vector<int> cache_lfu_count = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int cache_lfu_count_scale        = 0;

  int cache_set_coverage           = 64;
  std::vector<int64_t> raw_counter = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  int64_t start_gid                    = 0;
  int to_update_id_count               = 10;
  std::vector<int64_t> to_update_gids  = {3, 51, 20, 13, 60, 44, 37, 46, 12, 41};
  std::vector<int> to_update_inc_count = {1, 2, 1, 4, 5, 6, 3, 1, 1, 6};

  std::string name = "default";

  SingleCacheSetTestParam& Random(int base_scale, int coverage, int update_count)
  {
    EXPECT_LE(update_count, coverage);
    cache_set_coverage    = coverage;
    name                  = "random";
    cache_lfu_count_scale = base_scale;
    std::vector<int> lid_perm(coverage);
    for (int i = 0; i < coverage; i++)
      lid_perm[i] = i;
    std::shuffle(lid_perm.begin(), lid_perm.end(), std::mt19937(std::random_device()()));
    raw_counter.clear();
    raw_counter.resize(coverage, -1);
    for (int i = 0; i < 32 && i < coverage; i++) {
      cache_tag_lids[i]     = lid_perm[i];
      int mask              = std::experimental::randint(0, 3);
      cache_tag_valid[i]    = mask & 1;
      cache_tag_modified[i] = mask & 2;
      cache_lfu_count[i]    = std::experimental::randint(1, 16383);
      if (cache_tag_valid[i]) {
        raw_counter[cache_tag_lids[i]] =
          ((int64_t)cache_lfu_count[i] << cache_lfu_count_scale) +
          std::experimental::randint(0, (1 << cache_lfu_count_scale) - 1);
      }
    }
    for (int i = 0; i < coverage; i++) {
      if (raw_counter[i] == -1) {
        raw_counter[i] =
          std::experimental::randint<int64_t>(0, 1 << (14LL + (int64_t)cache_lfu_count_scale));
      }
    }
    start_gid          = std::experimental::randint<int64_t>(0, 1000000000LL) * coverage;
    to_update_id_count = update_count;
    to_update_gids.resize(to_update_id_count);
    to_update_inc_count.resize(to_update_id_count);
    std::shuffle(lid_perm.begin(), lid_perm.end(), std::mt19937(std::random_device()()));
    for (int i = 0; i < to_update_id_count; i++) {
      to_update_gids[i]      = lid_perm[i] + start_gid;
      to_update_inc_count[i] = std::experimental::randint(1, 16383);
    }
    return *this;
  }

  void Print() const
  {
    std::cout << "Test Parameter for " << name << ":" << std::endl;
    std::cout << "  " << "old_scale=" << cache_lfu_count_scale
              << ", cache_set_coverage=" << cache_set_coverage << ", start_gid=" << start_gid
              << ", to_update_id_count=" << to_update_id_count << std::endl;
    PrintVector(cache_tag_lids, "cache_tag_lids", 2);
    PrintVector(cache_tag_valid, "cache_tag_valid", 2);
    PrintVector(cache_tag_modified, "cache_tag_modified", 2);
    PrintVector(cache_lfu_count, "cache_lfu_count", 2);
    PrintVector(raw_counter, "raw_counter", 2);
    PrintVector(to_update_gids, "to_update_gids", 2);
    PrintVector(to_update_inc_count, "to_update_inc_count", 2);
  }
};

void PrintInfo(const SingleCacheSetTestParam& test_param,
               const std::vector<uint16_t>& cache_tag_vec_updated,
               const std::vector<uint16_t>& cache_lfu_vec_updated,
               const std::vector<int64_t>& raw_counter_updated,
               const std::vector<int64_t>& load_to_cache_ids,
               const std::vector<int64_t>& write_back_ids)
{
  std::cout << "\nTesting case " << test_param.name << std::endl;
  test_param.Print();
  std::cout << "*********************** Results ***********************" << std::endl;
  PrintTagVector(cache_tag_vec_updated, "cache_tag_vec_updated");
  PrintLfuCountVector(cache_lfu_vec_updated, "cache_lfu_vec_updated");
  PrintVector(raw_counter_updated, "raw_counter_updated");
  PrintVector(load_to_cache_ids, "load_to_cache_ids");
  PrintVector(write_back_ids, "write_back_ids");
}

bool CheckSingleResult(const SingleCacheSetTestParam& test_param,
                       const std::vector<uint16_t>& cache_tag_vec_updated,
                       const std::vector<uint16_t>& cache_lfu_vec_updated,
                       const std::vector<int64_t>& raw_counter_updated,
                       const std::vector<int64_t>& load_to_cache_ids,
                       const std::vector<int64_t>& write_back_ids)
{
  std::vector<int> updated_tag_lids(32);
  std::vector<bool> updated_valid(32);
  std::vector<bool> updated_modified(32);
  std::vector<int> updated_lfu_count(32);
  int update_scale = 0;

  EXPECT_EQ(cache_tag_vec_updated.size(), 32);
  EXPECT_EQ(cache_lfu_vec_updated.size(), 32);

  std::map<int, int> old_lid_to_cacheline;
  for (int i = 0; i < 32; i++) {
    if (test_param.cache_tag_valid[i]) {
      old_lid_to_cacheline.insert(std::pair<int, int>(test_param.cache_tag_lids[i], i));
    }
  }

  // extract results from cache data
  int in_cache_count = 0;
  std::set<int> updated_lid_dedup_set;
  for (int i = 0; i < cache_tag_vec_updated.size(); i++) {
    int lid    = cache_tag_vec_updated[i] & ((1 << 14) - 1);
    bool valid = cache_tag_vec_updated[i] & (1 << 14);
    if (!valid) lid = -1;
    bool modified = cache_tag_vec_updated[i] & (1 << 15);
    if (!valid) EXPECT_FALSE(modified);
    int lfu_count = cache_lfu_vec_updated[i] & ((1 << 14) - 1);
    if (!valid) lfu_count = -1;
    if ((cache_lfu_vec_updated[i] & (1 << 14)) != 0) { update_scale |= (1 << i); }
    if (valid) {
      EXPECT_EQ(updated_lid_dedup_set.find(lid), updated_lid_dedup_set.end());
      updated_lid_dedup_set.insert(lid);
      in_cache_count++;
    }

    updated_tag_lids[i]  = lid;
    updated_valid[i]     = valid;
    updated_modified[i]  = modified;
    updated_lfu_count[i] = lfu_count;
  }
  // Check ids in cache set are corrected
  std::map<int, int64_t> lid_to_count;
  std::map<int, int> base_lid_to_cache_line;
  for (int i = 0; i < 32; i++) {
    if (test_param.cache_tag_valid[i]) {
      EXPECT_GE(test_param.cache_tag_lids[i], 0);
      base_lid_to_cache_line.insert(std::pair<int, int>(test_param.cache_tag_lids[i], i));
      int lid       = test_param.cache_tag_lids[i];
      int64_t count = (static_cast<int64_t>(test_param.cache_lfu_count[i] + 1)
                       << test_param.cache_lfu_count_scale) -
                      1;
      lid_to_count[test_param.cache_tag_lids[i]] = count;
    }
  }
  std::vector<int64_t> ref_counter_updated = test_param.raw_counter;
  std::map<int, int> to_process_lid_to_array_id;
  for (int i = 0; i < test_param.to_update_id_count; i++) {
    int64_t gid   = test_param.to_update_gids[i];
    int lid       = gid - test_param.start_gid;
    int inc_count = test_param.to_update_inc_count[i];
    ref_counter_updated[lid] += inc_count;
    lid_to_count[lid] = ref_counter_updated[lid];
    to_process_lid_to_array_id.insert(std::pair<int, int>(lid, i));
  }
  std::vector<std::tuple<int64_t, int>> count_lid_vec;
  for (auto& lid_count : lid_to_count) {
    count_lid_vec.push_back(std::tuple<int64_t, int>(lid_count.second, lid_count.first));
  }
  std::sort(count_lid_vec.begin(), count_lid_vec.end(), std::greater{});
  int64_t max_lfu_count       = std::get<0>(count_lid_vec.front());
  int reference_updated_scale = 0;
  while ((max_lfu_count >> reference_updated_scale) >= (1 << 14)) {
    reference_updated_scale++;
  }
  EXPECT_EQ(reference_updated_scale, update_scale);
  // Compute reference ids
  std::set<int> must_in_cache, maybe_in_cache;
  int should_in_cache_count = 32;
  if (count_lid_vec.size() < 32) {
    for (auto& t : count_lid_vec)
      must_in_cache.insert(std::get<1>(t));
    should_in_cache_count = count_lid_vec.size();
  } else {
    int64_t min_count = std::get<0>(count_lid_vec[32 - 1]);
    for (auto& t : count_lid_vec) {
      if (std::get<0>(t) > min_count) {
        must_in_cache.insert(std::get<1>(t));
      } else {
        maybe_in_cache.insert(std::get<1>(t));
      }
    }
  }
  EXPECT_EQ(should_in_cache_count, in_cache_count);
  // Check all must_in_cache lids are in cache.
  for (auto must_in_lid : must_in_cache) {
    EXPECT_NE(updated_lid_dedup_set.find(must_in_lid), updated_lid_dedup_set.end());
  }
  // Check all cached elements are in must_in_cache or maybe_in_cache
  for (int i = 0; i < 32; i++) {
    if (!updated_valid[i]) continue;
    int lid = updated_tag_lids[i];
    EXPECT_TRUE(must_in_cache.find(lid) != must_in_cache.end() ||
                maybe_in_cache.find(lid) != maybe_in_cache.end());
    auto it = old_lid_to_cacheline.find(lid);
    // same lid won't change location.
    if (it != old_lid_to_cacheline.end()) {
      EXPECT_EQ(it->second, i);
      EXPECT_EQ(test_param.cache_tag_modified[i], updated_modified[i]);
      int64_t old_est_lfu_count =
        (((int64_t)test_param.cache_lfu_count[i] + 1) << test_param.cache_lfu_count_scale) - 1;
      auto it = to_process_lid_to_array_id.find(lid);
      if (it != to_process_lid_to_array_id.end()) {
        int64_t updated_est_lfu_count =
          test_param.raw_counter[lid] + test_param.to_update_inc_count[it->second];
        EXPECT_EQ(updated_est_lfu_count >> reference_updated_scale, updated_lfu_count[i])
          << "Index=" << i << ", lid=" << lid;
      } else {
        EXPECT_EQ(old_est_lfu_count >> reference_updated_scale, updated_lfu_count[i])
          << "Index=" << i << ", lid=" << lid;
      }
    } else {
      EXPECT_FALSE(updated_modified[i]) << "Index=" << i << ", lid=" << lid;
      auto it = to_process_lid_to_array_id.find(lid);
      EXPECT_NE(it, to_process_lid_to_array_id.end());
      int64_t updated_est_lfu_count =
        test_param.raw_counter[lid] + test_param.to_update_inc_count[it->second];
      EXPECT_EQ(updated_est_lfu_count >> reference_updated_scale, updated_lfu_count[i]);
    }
  }
  // Check all counters are right.
  for (int i = 0; i < test_param.cache_set_coverage; i++) {
    EXPECT_EQ(ref_counter_updated[i], raw_counter_updated[i]);
  }

  // load and writeback check
  int ld_count = 0, wb_count = 0;
  for (int i = 0; i < 32; i++) {
    if (updated_valid[i] &&
        (!test_param.cache_tag_valid[i] || test_param.cache_tag_lids[i] != updated_tag_lids[i])) {
      EXPECT_GT(load_to_cache_ids.size(), ld_count);
      EXPECT_EQ(load_to_cache_ids[ld_count], updated_tag_lids[i] + test_param.start_gid)
        << " index=" << ld_count;
      ld_count++;
    }
    if (test_param.cache_tag_valid[i] && test_param.cache_tag_modified[i] &&
        updated_tag_lids[i] != test_param.cache_tag_lids[i]) {
      EXPECT_GT(write_back_ids.size(), wb_count);
      EXPECT_EQ(write_back_ids[wb_count], test_param.cache_tag_lids[i] + test_param.start_gid)
        << " index=" << wb_count;
      wb_count++;
    }
  }
  if (testing::Test::HasFailure()) { return false; }
  return true;
}

class CacheSetSingleCaseTests : public ::testing::TestWithParam<SingleCacheSetTestParam> {};

__global__ void SingleCacheSetTestKernel(uint16_t* cache_set_tag_ptr,
                                         uint16_t* cache_set_count_ptr,
                                         int64_t* memory_lfu_counter,
                                         const int64_t* gids,
                                         const int* inc_count,
                                         int64_t* need_load_to_cache_ids,
                                         int64_t* need_write_back_ids,
                                         int64_t set_start_id,
                                         int id_count)
{
  using Updater = wholememory_ops::CacheSetUpdater<int64_t>;
  __shared__ Updater::TempStorage temp_storage;
  wholememory_ops::CacheLineInfo cache_line_info;
  cache_line_info.LoadInfo(cache_set_tag_ptr, cache_set_count_ptr);
  Updater updater;
  updater.UpdateCache<true, true>(temp_storage,
                                  cache_line_info,
                                  memory_lfu_counter,
                                  gids,
                                  inc_count,
                                  need_load_to_cache_ids,
                                  need_write_back_ids,
                                  set_start_id,
                                  id_count);
  cache_line_info.StoreInfo(cache_set_tag_ptr, cache_set_count_ptr);
}

TEST_P(CacheSetSingleCaseTests, CacheSetTest)
{
  static constexpr int kCacheSetSize = 32;
  auto params                        = GetParam();
  int dev_count;
  EXPECT_EQ(cudaGetDeviceCount(&dev_count), cudaSuccess);
  EXPECT_GE(dev_count, 1);
  EXPECT_EQ(cudaSetDevice(0), cudaSuccess);

  EXPECT_EQ(kCacheSetSize, params.cache_tag_lids.size());
  EXPECT_EQ(kCacheSetSize, params.cache_tag_valid.size());
  EXPECT_EQ(kCacheSetSize, params.cache_tag_modified.size());
  EXPECT_EQ(kCacheSetSize, params.cache_lfu_count.size());

  EXPECT_EQ(params.cache_set_coverage, params.raw_counter.size());

  EXPECT_EQ(params.to_update_id_count, params.to_update_gids.size());
  EXPECT_EQ(params.to_update_id_count, params.to_update_inc_count.size());

  uint16_t *cache_tag_ptr, *cache_lfu_ptr;
  EXPECT_EQ(cudaMalloc(&cache_tag_ptr, sizeof(uint16_t) * kCacheSetSize), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&cache_lfu_ptr, sizeof(uint16_t) * kCacheSetSize), cudaSuccess);

  int64_t *to_update_ids, *write_back_ids, *load_to_cache_ids;
  size_t update_ids_size = sizeof(int64_t) * params.to_update_id_count;
  EXPECT_EQ(cudaMalloc(&to_update_ids, update_ids_size), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&write_back_ids, update_ids_size), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&load_to_cache_ids, update_ids_size), cudaSuccess);

  int* inc_count;
  EXPECT_EQ(cudaMalloc(&inc_count, sizeof(int) * params.to_update_id_count), cudaSuccess);

  int64_t* raw_counter_ptr;
  EXPECT_EQ(cudaMalloc(&raw_counter_ptr, sizeof(int64_t) * params.cache_set_coverage), cudaSuccess);

  std::vector<int16_t> cache_tag_vec(kCacheSetSize, 0), cache_lfu_vec(kCacheSetSize, 0);
  for (int i = 0; i < kCacheSetSize; i++) {
    uint16_t tag_data          = params.cache_tag_valid[i] ? params.cache_tag_lids[i] : 0;
    uint16_t valid_modify_mask = 0;
    if (params.cache_tag_valid[i]) valid_modify_mask |= (1U << 14U);
    if (params.cache_tag_modified[i]) valid_modify_mask |= (1U << 15U);
    tag_data |= valid_modify_mask;
    cache_tag_vec[i] = tag_data;

    uint32_t scale          = params.cache_lfu_count_scale;
    uint16_t lfu_count_data = params.cache_lfu_count[i];
    if (scale & (1U << i)) { lfu_count_data |= (1U << 14U); }
    cache_lfu_vec[i] = lfu_count_data;
  }

  EXPECT_EQ(
    cudaMemcpy(
      cache_tag_ptr, cache_tag_vec.data(), kCacheSetSize * sizeof(int16_t), cudaMemcpyHostToDevice),
    cudaSuccess);
  EXPECT_EQ(
    cudaMemcpy(
      cache_lfu_ptr, cache_lfu_vec.data(), kCacheSetSize * sizeof(int16_t), cudaMemcpyHostToDevice),
    cudaSuccess);

  EXPECT_EQ(cudaMemcpy(raw_counter_ptr,
                       params.raw_counter.data(),
                       params.cache_set_coverage * sizeof(int64_t),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  EXPECT_EQ(cudaMemcpy(to_update_ids,
                       params.to_update_gids.data(),
                       params.to_update_id_count * sizeof(int64_t),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(inc_count,
                       params.to_update_inc_count.data(),
                       params.to_update_id_count * sizeof(int),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  SingleCacheSetTestKernel<<<1, kCacheSetSize>>>(cache_tag_ptr,
                                                 cache_lfu_ptr,
                                                 raw_counter_ptr,
                                                 to_update_ids,
                                                 inc_count,
                                                 load_to_cache_ids,
                                                 write_back_ids,
                                                 params.start_gid,
                                                 params.to_update_id_count);

  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<uint16_t> cache_tag_vec_updated(kCacheSetSize), cache_lfu_vec_updated(kCacheSetSize);
  std::vector<int64_t> raw_counter_updated(params.cache_set_coverage);
  std::vector<int64_t> load_to_cache_ids_vec(params.to_update_id_count),
    writeback_ids_vec(params.to_update_id_count);
  EXPECT_EQ(cudaMemcpy(cache_tag_vec_updated.data(),
                       cache_tag_ptr,
                       kCacheSetSize * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(cache_lfu_vec_updated.data(),
                       cache_lfu_ptr,
                       kCacheSetSize * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(raw_counter_updated.data(),
                       raw_counter_ptr,
                       params.cache_set_coverage * sizeof(int64_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(load_to_cache_ids_vec.data(),
                       load_to_cache_ids,
                       params.to_update_id_count * sizeof(int64_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(writeback_ids_vec.data(),
                       write_back_ids,
                       params.to_update_id_count * sizeof(int64_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  bool success = CheckSingleResult(params,
                                   cache_tag_vec_updated,
                                   cache_lfu_vec_updated,
                                   raw_counter_updated,
                                   load_to_cache_ids_vec,
                                   writeback_ids_vec);

  if (!success) {
    PrintInfo(params,
              cache_tag_vec_updated,
              cache_lfu_vec_updated,
              raw_counter_updated,
              load_to_cache_ids_vec,
              writeback_ids_vec);
  }

  EXPECT_EQ(cudaFree(cache_tag_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(cache_lfu_ptr), cudaSuccess);

  EXPECT_EQ(cudaFree(to_update_ids), cudaSuccess);
  EXPECT_EQ(cudaFree(write_back_ids), cudaSuccess);
  EXPECT_EQ(cudaFree(load_to_cache_ids), cudaSuccess);

  EXPECT_EQ(cudaFree(inc_count), cudaSuccess);
}

static SingleCacheSetTestParam modify_test{
  .cache_tag_lids     = {1, 3, 5, 9,  11, 13, 15, 17, 19, 21, 23, 25, 26, 28, 51, 60,
                         2, 4, 8, 10, 12, 14, 6,  18, 20, 22, 24, 0,  27, 40, 54, 62},
  .cache_tag_valid    = {true, true, true, true, true, true, true, true, true, true, true,
                         true, true, true, true, true, true, true, true, true, true, true,
                         true, true, true, true, true, true, true, true, true, true},
  .cache_tag_modified = {false, true,  false, true, false, true,  false, false, false, true, true,
                         false, true,  true,  true, false, false, false, true,  false, true, false,
                         true,  false, false, true, false, true,  true,  true,  true,  false},
  .cache_lfu_count    = {1023, 435, 435, 23,   981, 13012, 231,  523,  1227, 8005, 324,
                         328,  443, 134, 218,  435, 32,    1324, 1112, 98,   1021, 1992,
                         4032, 747, 382, 1211, 832, 5123,  56,   1212, 622,  646},
  .cache_lfu_count_scale = 0,
  .cache_set_coverage    = 64,
  .raw_counter = {5123,  1023, 32,  435,  1324, 435,  4032, 245,  1112, 23,   98,   981,  1021,
                  13012, 1992, 231, 383,  523,  747,  1227, 382,  8005, 1211, 324,  832,  328,
                  443,   56,   134, 543,  998,  1768, 1211, 1321, 223,  148,  1234, 1211, 832,
                  1437,  1212, 82,  1080, 345,  1643, 1432, 424,  567,  839,  911,  493,  218,
                  1821,  921,  622, 1718, 428,  1283, 1198, 2355, 435,  864,  646,  346},
  .start_gid   = 128,
  .to_update_id_count  = 10,
  .to_update_gids      = {129, 157, 141, 177, 149, 182, 138, 191, 134, 144},
  .to_update_inc_count = {53, 128, 4323, 321, 493, 232, 98, 43, 22, 134},
  .name                = "modify_test"};

static SingleCacheSetTestParam small_test{
  .cache_tag_lids  = {0,  -1, -1, -1, -1, 2,  -1, -1, -1, 3,  -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, 0,  -1, -1, -1, -1},
  .cache_tag_valid = {false, false, false, false, false, true,  false, false, false, true,  false,
                      false, false, false, false, false, false, false, false, false, false, false,
                      true,  false, false, false, false, true,  false, false, false, false},
  .cache_tag_modified    = {false, false, false, false, false, true,  false, false,
                            false, false, false, false, false, false, false, false,
                            false, false, false, false, false, false, false, false,
                            false, false, false, true,  false, false, false, false},
  .cache_lfu_count       = {0, 0, 0, 0, 0, 13012, 0,    0, 0, 2005, 0, 0,    0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,     4032, 0, 0, 0,    0, 5123, 0, 0, 0, 0},
  .cache_lfu_count_scale = 0,
  .cache_set_coverage    = 5,
  .raw_counter           = {5123, 4032, 13012, 2005, 1324},
  .start_gid             = 100,
  .to_update_id_count    = 3,
  .to_update_gids        = {100, 103, 104},
  .to_update_inc_count   = {53, 128, 4323},
  .name                  = "small_test"};

static SingleCacheSetTestParam medium_test{
  .cache_tag_lids  = {0,  0,  0,  -1, -1, 13, -1, -1, -1, 21, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, 6,  -1, -1, -1, -1, 0,  -1, -1, -1, -1},
  .cache_tag_valid = {false, false, false, false, false, true,  false, false, false, true,  false,
                      false, false, false, false, false, false, false, false, false, false, false,
                      true,  false, false, false, false, true,  false, false, false, false},
  .cache_tag_modified    = {false, false, false, false, false, true,  false, false,
                            false, false, false, false, false, false, false, false,
                            false, false, false, false, false, false, false, false,
                            false, false, false, true,  false, false, false, false},
  .cache_lfu_count       = {0, 0, 0, 0, 0, 13012, 0,    0, 0, 2005, 0, 0,    0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,     4032, 0, 0, 0,    0, 5123, 0, 0, 0, 0},
  .cache_lfu_count_scale = 0,
  .cache_set_coverage    = 64,
  .raw_counter = {5123,  1023, 32,  435,  1324, 435,  4032, 245,  1112, 23,   98,   981,  1021,
                  13012, 1992, 231, 383,  523,  747,  1227, 382,  8005, 1211, 324,  832,  328,
                  443,   56,   134, 543,  998,  1768, 1211, 1321, 223,  148,  1234, 1211, 832,
                  1437,  1212, 82,  1080, 345,  1643, 1432, 424,  567,  839,  911,  493,  218,
                  1821,  921,  622, 1718, 428,  1283, 1198, 2355, 435,  864,  646,  346},
  .start_gid   = 128,
  .to_update_id_count  = 10,
  .to_update_gids      = {129, 157, 141, 177, 149, 182, 138, 191, 134, 144},
  .to_update_inc_count = {53, 128, 4323, 321, 493, 232, 98, 43, 22, 134},
  .name                = "medium_test"};

static SingleCacheSetTestParam large_test{
  .cache_tag_lids     = {1, 3, 5, 9,  11, 13, 15, 17, 19, 21, 23, 25, 26, 28, 51, 60,
                         2, 4, 8, 10, 12, 14, 6,  18, 20, 22, 24, 0,  27, 40, 54, 62},
  .cache_tag_valid    = {true, true, true, true, true, true, true, true, true, true, true,
                         true, true, true, true, true, true, true, true, true, true, true,
                         true, true, true, true, true, true, true, true, true, true},
  .cache_tag_modified = {true,  false, false, true, false, true,  false, false, true,  true, true,
                         false, true,  true,  true, false, false, false, true,  false, true, false,
                         true,  false, false, true, false, true,  true,  true,  true,  false},
  .cache_lfu_count = {16052, 15443, 14375, 12384, 12981, 13012, 15231, 15523, 14323, 10212, 13101,
                      12320, 10443, 14134, 16118, 14350, 14935, 13244, 11112, 11098, 11021, 11992,
                      14032, 13747, 13387, 12117, 11832, 15123, 15456, 13212, 13622, 11646},
  .cache_lfu_count_scale = 0,
  .cache_set_coverage    = 564,
  .raw_counter =
    {5123, 15443, 32,   435,  1324, 435,  4032, 245,  1112, 23,   98,   981,  1021, 13012, 1992,
     231,  383,   523,  747,  1227, 382,  8005, 1211, 324,  832,  328,  443,  56,   134,   543,
     998,  1768,  1211, 1321, 223,  148,  1234, 1211, 832,  1437, 1212, 82,   1080, 345,   1643,
     1432, 424,   567,  839,  911,  493,  218,  1821, 921,  622,  1718, 428,  1283, 1198,  2355,
     435,  864,   646,  346,  5648, 4585, 8582, 719,  2589, 8971, 2506, 8304, 1806, 227,   8888,
     1768, 2639,  39,   3973, 459,  4894, 5951, 3781, 4742, 1164, 1816, 7519, 8851, 9220,  2633,
     2190, 786,   3953, 6205, 1906, 5076, 5105, 4325, 7744, 3166, 8925, 1643, 7225, 8278,  7081,
     2416, 1508,  3839, 6741, 9856, 6771, 9414, 7760, 3782, 882,  8318, 3215, 2138, 4395,  8355,
     731,  7970,  5753, 6595, 330,  7038, 4197, 1208, 7511, 6385, 4576, 7625, 8684, 522,   1363,
     3336, 2904,  286,  2893, 4189, 8698, 7908, 7670, 6578, 5431, 8368, 8182, 6825, 1831,  6060,
     2107, 1196,  2364, 2289, 4702, 6030, 5318, 4248, 1554, 3205, 8072, 3107, 3334, 3824,  7651,
     5500, 2051,  6175, 9526, 9888, 253,  6108, 4949, 4317, 485,  4694, 9210, 1214, 4878,  6513,
     6727, 85,    8023, 691,  794,  1345, 5172, 7539, 4341, 566,  5865, 4521, 5674, 6127,  3323,
     6815, 6906,  6668, 2588, 3545, 3480, 6000, 7982, 633,  1353, 7631, 248,  4287, 1870,  8083,
     5187, 1994,  9851, 2221, 4809, 5023, 1057, 7305, 6315, 837,  7281, 1743, 5507, 9712,  5893,
     5959, 7229,  7949, 1656, 8973, 7083, 8529, 7699, 2690, 9797, 2200, 4344, 6601, 1239,  6522,
     4689, 2285,  6114, 8070, 1398, 9452, 7836, 5294, 6305, 4656, 804,  6852, 4307, 4720,  9817,
     2964, 2327,  5876, 9681, 8494, 7591, 9407, 1599, 3805, 4251, 8182, 5127, 5512, 7352,  1761,
     4268, 1220,  4626, 6586, 8623, 8392, 1698, 7226, 4468, 5713, 9219, 176,  2160, 552,   8808,
     9251, 2118,  9069, 7314, 7776, 3941, 3027, 1670, 3710, 2968, 4877, 6135, 2737, 732,   1681,
     9418, 4751,  9061, 8579, 9087, 917,  7513, 2906, 1168, 4364, 9368, 4622, 3492, 2210,  505,
     1779, 1494,  519,  4478, 7958, 9938, 6636, 6081, 8852, 8518, 3094, 5616, 4802, 7196,  332,
     7204, 2291,  2626, 1247, 6454, 7642, 1189, 4454, 4819, 9813, 9771, 7137, 4721, 9919,  8594,
     8443, 8574,  5554, 3104, 1200, 3118, 9787, 9503, 5527, 3047, 3231, 5714, 4561, 9928,  3954,
     4496, 8760,  1700, 1984, 9969, 2750, 5143, 2176, 2012, 7975, 841,  1727, 6338, 7469,  9428,
     5328, 8245,  8744, 1169, 9707, 5258, 3199, 1288, 3505, 5490, 2387, 7191, 3003, 6768,  7669,
     3634, 1037,  6794, 7834, 174,  3692, 7498, 6992, 8051, 2786, 759,  4912, 5601, 9623,  529,
     411,  9399,  1185, 6379, 2103, 165,  3169, 2145, 5374, 3762, 2752, 8824, 8635, 3801,  5684,
     490,  4675,  3574, 8381, 4193, 3081, 3374, 5280, 5990, 2192, 1754, 1209, 6229, 1927,  9347,
     2760, 7652,  1500, 7904, 4155, 6215, 4964, 2274, 4844, 8865, 4909, 6082, 5330, 9388,  5218,
     2322, 5076,  1342, 4081, 9607, 1181, 2113, 3462, 8662, 6134, 5903, 7965, 5581, 9891,  8705,
     5451, 1289,  3383, 3968, 4389, 3168, 922,  9339, 7622, 9600, 5754, 3631, 8317, 3698,  4132,
     241,  4686,  4280, 149,  9151, 7115, 2583, 2077, 2544, 9998, 2736, 8915, 3767, 3640,  360,
     5059, 7880,  921,  8832, 9432, 6848, 5673, 1247, 992,  5988, 3913, 2010, 4496, 9034,  844,
     8482, 5071,  6610, 750,  9525, 8782, 3453, 5300, 6955, 6986, 678,  5760, 3055, 3352,  7631,
     5751, 8796,  9636, 794,  7302, 5804, 2542, 6996, 4498, 6318, 443,  2100, 732,  302,   6633,
     8169, 2662,  3265, 7489, 6825, 545,  4222, 2936, 5612, 2039, 8772, 5632, 7284, 679,   7634,
     8935, 9315,  8439, 3390, 3,    9367, 3139, 8860, 9408},
  .start_gid          = 564 * 2,
  .to_update_id_count = 301,
  .to_update_gids =
    {1385, 1257, 1684, 1341, 1302, 1550, 1536, 1665, 1233, 1259, 1452, 1574, 1219, 1637, 1633, 1155,
     1660, 1492, 1273, 1612, 1380, 1329, 1650, 1382, 1628, 1520, 1150, 1429, 1526, 1264, 1327, 1339,
     1600, 1575, 1506, 1147, 1538, 1358, 1681, 1497, 1365, 1511, 1471, 1209, 1253, 1395, 1611, 1516,
     1143, 1229, 1335, 1552, 1639, 1479, 1180, 1352, 1654, 1541, 1314, 1465, 1437, 1469, 1585, 1203,
     1666, 1629, 1376, 1576, 1474, 1261, 1610, 1608, 1525, 1514, 1216, 1188, 1231, 1675, 1196, 1531,
     1445, 1330, 1548, 1553, 1599, 1535, 1243, 1360, 1417, 1284, 1190, 1510, 1357, 1425, 1228, 1185,
     1647, 1502, 1463, 1217, 1234, 1581, 1508, 1202, 1244, 1554, 1447, 1411, 1649, 1591, 1683, 1448,
     1475, 1348, 1232, 1182, 1305, 1459, 1495, 1635, 1204, 1603, 1258, 1609, 1291, 1439, 1604, 1205,
     1211, 1389, 1183, 1159, 1331, 1176, 1394, 1384, 1595, 1632, 1252, 1313, 1221, 1444, 1315, 1563,
     1499, 1436, 1157, 1407, 1614, 1387, 1539, 1274, 1300, 1468, 1248, 1263, 1615, 1236, 1435, 1476,
     1513, 1173, 1621, 1477, 1301, 1139, 1207, 1379, 1423, 1277, 1657, 1344, 1561, 1132, 1688, 1247,
     1587, 1646, 1582, 1168, 1337, 1212, 1518, 1651, 1359, 1144, 1289, 1275, 1442, 1386, 1171, 1398,
     1605, 1408, 1456, 1601, 1362, 1590, 1392, 1427, 1640, 1478, 1440, 1317, 1562, 1524, 1679, 1593,
     1371, 1161, 1287, 1271, 1432, 1509, 1268, 1377, 1586, 1519, 1512, 1312, 1624, 1424, 1449, 1673,
     1662, 1641, 1189, 1528, 1588, 1627, 1356, 1664, 1149, 1288, 1682, 1517, 1145, 1278, 1192, 1251,
     1332, 1197, 1668, 1181, 1151, 1397, 1272, 1129, 1403, 1396, 1690, 1626, 1458, 1678, 1622, 1613,
     1466, 1464, 1462, 1166, 1472, 1606, 1616, 1152, 1280, 1583, 1428, 1255, 1167, 1138, 1319, 1325,
     1195, 1170, 1529, 1299, 1555, 1617, 1318, 1584, 1638, 1153, 1201, 1218, 1186, 1671, 1369, 1158,
     1308, 1136, 1294, 1283, 1661, 1148, 1547, 1311, 1163, 1540, 1644, 1433, 1326},
  .to_update_inc_count =
    {5981, 9042, 973,  2226, 854,  852,  7049, 7321, 8050, 5163, 6911, 982,  9499, 1054, 4416, 2165,
     2370, 368,  6701, 1535, 4436, 5248, 8373, 413,  9714, 4219, 4004, 1499, 1869, 2981, 8502, 7151,
     9540, 2599, 675,  9694, 9449, 7763, 340,  8713, 3265, 6000, 457,  3544, 3728, 8080, 885,  212,
     5650, 936,  6505, 8174, 1289, 6716, 2555, 5105, 7914, 7139, 4365, 867,  3965, 9160, 4308, 8185,
     6733, 295,  5703, 1233, 1911, 8645, 5491, 290,  9863, 7433, 4131, 7492, 9820, 6942, 4029, 5172,
     7543, 2824, 7025, 3613, 3949, 2460, 8218, 8850, 1318, 5066, 3412, 4549, 2381, 6508, 2363, 5036,
     7648, 8770, 7072, 1420, 3327, 1256, 7764, 7754, 9291, 1269, 536,  6053, 3986, 8556, 2952, 46,
     1168, 201,  8149, 6083, 9268, 1812, 8588, 4570, 4264, 879,  2978, 640,  3426, 4150, 3856, 4796,
     8111, 107,  3856, 921,  7074, 9830, 6834, 8179, 2449, 8987, 433,  5333, 3101, 4997, 1702, 6988,
     8137, 3829, 4773, 8572, 600,  9491, 6046, 3345, 7287, 6231, 1987, 4090, 3120, 4693, 3296, 3413,
     8912, 6731, 8819, 3574, 5593, 1252, 5849, 8096, 9798, 2682, 180,  1082, 8307, 66,   8985, 4710,
     8863, 5113, 6863, 1711, 470,  455,  920,  8954, 7635, 4959, 2015, 9372, 5116, 1952, 4506, 7781,
     5440, 5517, 1597, 4127, 2295, 8661, 5313, 2370, 1825, 8812, 973,  9608, 6754, 6232, 644,  7299,
     2651, 6003, 961,  927,  1176, 4088, 6704, 832,  919,  9768, 9495, 9495, 8424, 7835, 5242, 6485,
     4592, 1914, 1801, 4748, 9097, 725,  7925, 2211, 6742, 1614, 8052, 8012, 9887, 6160, 9161, 5545,
     9814, 3598, 938,  6706, 3872, 1932, 4630, 5308, 487,  7306, 2870, 6778, 4684, 155,  8430, 1234,
     3048, 1874, 4456, 2971, 2954, 3132, 3006, 4403, 5876, 1131, 5834, 2547, 4051, 772,  5353, 5751,
     968,  3021, 1402, 4198, 98,   7635, 4559, 3067, 8476, 8574, 5676, 1766, 8834, 5284, 7752, 7682,
     920,  4078, 4887, 1333, 5714, 7995, 6125, 1293, 2299, 5103, 7710, 3755, 4399},
  .name = "large_test"};

static SingleCacheSetTestParam big_number_test{
  .cache_tag_lids =
    {
      3,  16, 6,  21, 10, 13, 9,  17, 24, 30, 27, 7, 28, 1,  5,  18,
      11, 12, 22, 4,  25, 8,  23, 20, 15, 32, 31, 2, 0,  19, 29, 26,
    },
  .cache_tag_valid    = {true,  true,  false, true, false, false, false, true,  true, false, false,
                         false, true,  false, true, true,  true,  true,  false, true, true,  true,
                         false, false, true,  true, false, true,  false, false, true, false},
  .cache_tag_modified = {false, true,  false, false, false, true,  false, false, true, false, true,
                         true,  false, false, true,  false, false, true,  false, true, true,  false,
                         true,  true,  false, false, true,  true,  true,  true,  true, false},
  .cache_lfu_count    = {7769,  7418, 15671, 10379, 12640, 4833, 2127, 1920,  9986,  3516, 9790,
                         15974, 3257, 5591,  487,   3603,  5892, 2805, 12370, 10538, 1750, 1011,
                         11314, 1826, 6522,  12076, 14259, 394,  1729, 11217, 12869, 15354},
  .cache_lfu_count_scale = 0,
  .cache_set_coverage    = 33,
  .raw_counter         = {747,  2510,  394,  7769, 10538, 487,   10896, 1907,  1011,  318,   13798,
                          5892, 2805,  7616, 4431, 6522,  7418,  1920,  3603,  10777, 8169,  10379,
                          9185, 15366, 9986, 1750, 12121, 12857, 3257,  12869, 6256,  11222, 12076},
  .start_gid           = 10375660383LL,
  .to_update_id_count  = 31,
  .to_update_gids      = {10375660412LL, 10375660389LL, 10375660395LL, 10375660410LL, 10375660396LL,
                          10375660402LL, 10375660409LL, 10375660390LL, 10375660384LL, 10375660388LL,
                          10375660408LL, 10375660401LL, 10375660415LL, 10375660394LL, 10375660407LL,
                          10375660386LL, 10375660387LL, 10375660399LL, 10375660383LL, 10375660414LL,
                          10375660392LL, 10375660393LL, 10375660406LL, 10375660398LL, 10375660385LL,
                          10375660397LL, 10375660400LL, 10375660413LL, 10375660403LL, 10375660391LL,
                          10375660405LL},
  .to_update_inc_count = {7199, 7919,  4122,  10946, 11288, 16077, 5925, 5390,  6516, 11525, 3016,
                          514,  1008,  12538, 14024, 13003, 12927, 4785, 10512, 830,  13930, 9032,
                          3992, 13998, 5678,  3930,  6631,  13014, 3571, 1769,  14647},
  .name                = "big_number_test"};

INSTANTIATE_TEST_SUITE_P(CacheSetTest,
                         CacheSetSingleCaseTests,
                         ::testing::Values(modify_test,
                                           big_number_test,
                                           small_test,
                                           medium_test,
                                           large_test,

                                           SingleCacheSetTestParam().Random(21, 12, 10),
                                           SingleCacheSetTestParam().Random(1, 11, 10),
                                           SingleCacheSetTestParam().Random(11, 10, 10),
                                           SingleCacheSetTestParam().Random(5, 9, 7),
                                           SingleCacheSetTestParam().Random(4, 8, 8),
                                           SingleCacheSetTestParam().Random(24, 7, 5),
                                           SingleCacheSetTestParam().Random(15, 6, 3),
                                           SingleCacheSetTestParam().Random(35, 5, 5),
                                           SingleCacheSetTestParam().Random(27, 4, 4),
                                           SingleCacheSetTestParam().Random(23, 3, 1),
                                           SingleCacheSetTestParam().Random(23, 2, 2),
                                           SingleCacheSetTestParam().Random(23, 1, 1),

                                           SingleCacheSetTestParam().Random(21, 11201, 948),
                                           SingleCacheSetTestParam().Random(0, 123, 31),
                                           SingleCacheSetTestParam().Random(0, 1212, 1002),
                                           SingleCacheSetTestParam().Random(0, 523, 523),
                                           SingleCacheSetTestParam().Random(0, 1001, 3),
                                           SingleCacheSetTestParam().Random(0, 33, 31),
                                           SingleCacheSetTestParam().Random(1, 542, 201),
                                           SingleCacheSetTestParam().Random(5, 123, 31),
                                           SingleCacheSetTestParam().Random(3, 16384, 5000),
                                           SingleCacheSetTestParam().Random(2, 10210, 432),
                                           SingleCacheSetTestParam().Random(10, 15422, 4392),
                                           SingleCacheSetTestParam().Random(11, 9382, 9382),
                                           SingleCacheSetTestParam().Random(6, 8437, 7983),
                                           SingleCacheSetTestParam().Random(18, 832, 38),
                                           SingleCacheSetTestParam().Random(32, 1121, 998),
                                           SingleCacheSetTestParam().Random(35, 3232, 99),
                                           SingleCacheSetTestParam().Random(41, 5242, 422),
                                           SingleCacheSetTestParam().Random(32, 292, 127),
                                           SingleCacheSetTestParam().Random(2, 948, 91),
                                           SingleCacheSetTestParam().Random(11, 3221, 942),
                                           SingleCacheSetTestParam().Random(22, 938, 150),

                                           SingleCacheSetTestParam().Random(21, 12, 10),
                                           SingleCacheSetTestParam().Random(1, 11, 10),
                                           SingleCacheSetTestParam().Random(11, 10, 10),
                                           SingleCacheSetTestParam().Random(5, 9, 7),
                                           SingleCacheSetTestParam().Random(4, 8, 8),
                                           SingleCacheSetTestParam().Random(24, 7, 5),
                                           SingleCacheSetTestParam().Random(15, 6, 3),
                                           SingleCacheSetTestParam().Random(35, 5, 5),
                                           SingleCacheSetTestParam().Random(27, 4, 4),
                                           SingleCacheSetTestParam().Random(23, 3, 1),
                                           SingleCacheSetTestParam().Random(23, 2, 2),
                                           SingleCacheSetTestParam().Random(23, 1, 1),

                                           SingleCacheSetTestParam().Random(21, 11201, 948),
                                           SingleCacheSetTestParam().Random(0, 123, 31),
                                           SingleCacheSetTestParam().Random(0, 1212, 1002),
                                           SingleCacheSetTestParam().Random(0, 523, 523),
                                           SingleCacheSetTestParam().Random(0, 1001, 3),
                                           SingleCacheSetTestParam().Random(0, 33, 31),
                                           SingleCacheSetTestParam().Random(1, 542, 201),
                                           SingleCacheSetTestParam().Random(5, 123, 31),
                                           SingleCacheSetTestParam().Random(3, 16384, 5000),
                                           SingleCacheSetTestParam().Random(2, 10210, 432),
                                           SingleCacheSetTestParam().Random(10, 15422, 4392),
                                           SingleCacheSetTestParam().Random(11, 9382, 9382),
                                           SingleCacheSetTestParam().Random(6, 8437, 7983),
                                           SingleCacheSetTestParam().Random(18, 832, 38),
                                           SingleCacheSetTestParam().Random(32, 1121, 998),
                                           SingleCacheSetTestParam().Random(35, 3232, 99),
                                           SingleCacheSetTestParam().Random(41, 5242, 422),
                                           SingleCacheSetTestParam().Random(32, 292, 127),
                                           SingleCacheSetTestParam().Random(2, 948, 91),
                                           SingleCacheSetTestParam().Random(11, 3221, 942),
                                           SingleCacheSetTestParam().Random(22, 938, 150),

                                           SingleCacheSetTestParam().Random(21, 12, 10),
                                           SingleCacheSetTestParam().Random(1, 11, 10),
                                           SingleCacheSetTestParam().Random(11, 10, 10),
                                           SingleCacheSetTestParam().Random(5, 9, 7),
                                           SingleCacheSetTestParam().Random(4, 8, 8),
                                           SingleCacheSetTestParam().Random(24, 7, 5),
                                           SingleCacheSetTestParam().Random(15, 6, 3),
                                           SingleCacheSetTestParam().Random(35, 5, 5),
                                           SingleCacheSetTestParam().Random(27, 4, 4),
                                           SingleCacheSetTestParam().Random(23, 3, 1),
                                           SingleCacheSetTestParam().Random(23, 2, 2),
                                           SingleCacheSetTestParam().Random(23, 1, 1),

                                           SingleCacheSetTestParam().Random(21, 11201, 948),
                                           SingleCacheSetTestParam().Random(0, 123, 31),
                                           SingleCacheSetTestParam().Random(0, 1212, 1002),
                                           SingleCacheSetTestParam().Random(0, 523, 523),
                                           SingleCacheSetTestParam().Random(0, 1001, 3),
                                           SingleCacheSetTestParam().Random(0, 33, 31),
                                           SingleCacheSetTestParam().Random(1, 542, 201),
                                           SingleCacheSetTestParam().Random(5, 123, 31),
                                           SingleCacheSetTestParam().Random(3, 16384, 5000),
                                           SingleCacheSetTestParam().Random(2, 10210, 432),
                                           SingleCacheSetTestParam().Random(10, 15422, 4392),
                                           SingleCacheSetTestParam().Random(11, 9382, 9382),
                                           SingleCacheSetTestParam().Random(6, 8437, 7983),
                                           SingleCacheSetTestParam().Random(18, 832, 38),
                                           SingleCacheSetTestParam().Random(32, 1121, 998),
                                           SingleCacheSetTestParam().Random(35, 3232, 99),
                                           SingleCacheSetTestParam().Random(41, 5242, 422),
                                           SingleCacheSetTestParam().Random(32, 292, 127),
                                           SingleCacheSetTestParam().Random(2, 948, 91),
                                           SingleCacheSetTestParam().Random(11, 3221, 942),
                                           SingleCacheSetTestParam().Random(22, 938, 150),

                                           SingleCacheSetTestParam().Random(21, 12, 10),
                                           SingleCacheSetTestParam().Random(1, 11, 10),
                                           SingleCacheSetTestParam().Random(11, 10, 10),
                                           SingleCacheSetTestParam().Random(5, 9, 7),
                                           SingleCacheSetTestParam().Random(4, 8, 8),
                                           SingleCacheSetTestParam().Random(24, 7, 5),
                                           SingleCacheSetTestParam().Random(15, 6, 3),
                                           SingleCacheSetTestParam().Random(35, 5, 5),
                                           SingleCacheSetTestParam().Random(27, 4, 4),
                                           SingleCacheSetTestParam().Random(23, 3, 1),
                                           SingleCacheSetTestParam().Random(23, 2, 2),
                                           SingleCacheSetTestParam().Random(23, 1, 1),

                                           SingleCacheSetTestParam().Random(21, 11201, 948),
                                           SingleCacheSetTestParam().Random(0, 123, 31),
                                           SingleCacheSetTestParam().Random(0, 1212, 1002),
                                           SingleCacheSetTestParam().Random(0, 523, 523),
                                           SingleCacheSetTestParam().Random(0, 1001, 3),
                                           SingleCacheSetTestParam().Random(0, 33, 31),
                                           SingleCacheSetTestParam().Random(1, 542, 201),
                                           SingleCacheSetTestParam().Random(5, 123, 31),
                                           SingleCacheSetTestParam().Random(3, 16384, 5000),
                                           SingleCacheSetTestParam().Random(2, 10210, 432),
                                           SingleCacheSetTestParam().Random(10, 15422, 4392),
                                           SingleCacheSetTestParam().Random(11, 9382, 9382),
                                           SingleCacheSetTestParam().Random(6, 8437, 7983),
                                           SingleCacheSetTestParam().Random(18, 832, 38),
                                           SingleCacheSetTestParam().Random(32, 1121, 998),
                                           SingleCacheSetTestParam().Random(35, 3232, 99),
                                           SingleCacheSetTestParam().Random(41, 5242, 422),
                                           SingleCacheSetTestParam().Random(32, 292, 127),
                                           SingleCacheSetTestParam().Random(2, 948, 91),
                                           SingleCacheSetTestParam().Random(11, 3221, 942),
                                           SingleCacheSetTestParam().Random(22, 938, 150),

                                           SingleCacheSetTestParam()));
