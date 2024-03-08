/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include "file_io.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "communicator.hpp"
#include "error.hpp"
#include "integer_utils.hpp"
#include "logger.hpp"

namespace wholememory {

static bool IsFileExist(const char* filename, int mode) { return access(filename, mode) == 0; }

static size_t StatFileSize(const char* filename)
{
  auto filesize = static_cast<size_t>(-1);
  struct stat statbuf {};
  if (stat(filename, &statbuf) < 0) { return filesize; }
  filesize = statbuf.st_size;
  return filesize;
}

static size_t StatFileBlockSize(const char* filename)
{
  auto blocksize = static_cast<size_t>(-1);
  struct stat statbuf {};
  if (stat(filename, &statbuf) < 0) { return blocksize; }
  blocksize = statbuf.st_blksize;
  return blocksize;
}

static size_t get_handle_partial_size(size_t handle_size,
                                      size_t memory_offset,
                                      size_t memory_entry_stride,
                                      size_t entry_size)
{
  handle_size -= memory_offset;
  size_t tail = handle_size % memory_entry_stride;
  if (tail != 0 && tail < entry_size) {
    WHOLEMEMORY_FAIL_NOTHROW(
      "handle_size=%ld, memory_offset=%ld, memory_entry_stride=%ld, entry_size=%ld, tail=%ld is "
      "not 0"
      " or >= entry_size.",
      handle_size,
      memory_offset,
      memory_entry_stride,
      entry_size,
      tail);
  }
  size_t partial_size = 0;
  if (tail != 0) partial_size = entry_size;
  partial_size += (handle_size / memory_entry_stride) * entry_size;
  return partial_size;
}

/*!
 * Read from file list to local memory of WholeMemory. File list are binary files, which are
 * considered to be concatenated together. All ranks in WholeMemory will read the files in parallel
 * and load each part into local memory of each rank. WholeMemory will use round-robin sharding
 * strategy.
 * @param local_ptr : Pointer to local memory of WholeMemory
 * @param local_size : Local memory size
 * @param local_offset : The offset of local memory in WholeMemory.
 * @param entry_size : The entry size of each data entry.
 * @param memory_entry_stride : The stride of each entry in WholeMemory
 * @param memory_offset : The start offset to place the read data. Should be in range [0,
 * memory_entry_stride)
 * @param file_count : Total file count of the file list
 * @param file_names : File names of the file list.
 * @param file_sizes : Sizes of each file.
 * @param suggested_buffer_size : Suggested buffer size to read.
 * @param wm_rank : WholeMemory rank.
 * @param wm_world_size : WholeMemory world size.
 * @param round_robin_size : continuous embedding size of a rank using round robin shard stratehy.
 */
static void read_file_list_to_local_memory_roundrobin(char* local_ptr,
                                                      size_t local_size,
                                                      size_t local_offset,
                                                      size_t entry_size,
                                                      size_t memory_entry_stride,
                                                      size_t memory_offset,
                                                      int file_count,
                                                      const char** file_names,
                                                      const std::vector<size_t>& file_sizes,
                                                      size_t suggested_buffer_size,
                                                      int wm_rank,
                                                      int wm_world_size,
                                                      int round_robin_size)
{
  size_t buffer_size;
  size_t buffer_entry_count = 1;
  if (suggested_buffer_size < entry_size) {
    buffer_size = entry_size;
  } else {
    buffer_entry_count = suggested_buffer_size / entry_size;
    buffer_size        = buffer_entry_count * entry_size;
  }
  std::vector<char> file_read_buffer(buffer_size);

  if (memory_offset >= memory_entry_stride)
    WHOLEMEMORY_ERROR("memory offset %lu should be less than memory entry stride %lu.",
                      memory_offset,
                      memory_entry_stride);
  size_t total_file_sizes = 0;
  for (int i = 0; i < file_count; i++)
    total_file_sizes += file_sizes[i];
  size_t total_file_entry_count = total_file_sizes / entry_size;
  if (round_robin_size <= 0 || round_robin_size > total_file_entry_count / wm_world_size)
    WHOLEMEMORY_ERROR("illegal round_robin_size.");

  size_t local_entry_memory_start_index = wm_rank * round_robin_size;
  size_t local_entry_file_start_index =
    local_entry_memory_start_index - memory_offset / memory_entry_stride;

  size_t extra_entry       = total_file_entry_count % (wm_world_size * round_robin_size);
  size_t local_extra_entry = (extra_entry > (wm_rank + 1) * round_robin_size)
                               ? round_robin_size
                               : extra_entry - wm_rank * round_robin_size;
  local_extra_entry        = local_extra_entry > 0 ? local_extra_entry : 0;
  size_t local_entry_count =
    total_file_entry_count / (wm_world_size * round_robin_size) * round_robin_size;
  local_entry_count += local_extra_entry;

  char* local_write_ptr = local_ptr + memory_offset % memory_entry_stride;
  if (wm_rank == 0) {
    local_entry_count -= memory_offset / memory_entry_stride;
    local_write_ptr += (memory_offset / memory_entry_stride) * memory_entry_stride;
  }

  size_t total_read_entry            = 0;
  size_t next_entry_gap              = local_entry_file_start_index;
  size_t next_continuous_entry_count = round_robin_size > local_entry_count - total_read_entry
                                         ? local_entry_count - total_read_entry
                                         : round_robin_size;
  size_t read_file_begin_entry_off   = 0;
  for (int i = 0; i < file_count; i++) {
    size_t file_entry_count = file_sizes[i] / entry_size;
    if (file_entry_count <= next_entry_gap) {
      next_entry_gap -= file_entry_count;
      continue;
    }
    size_t read_size_from_cur_file = 0;
    read_file_begin_entry_off      = 0;
    //$open file get fp
    FILE* fp = fopen(file_names[i], "rb");
    if (fp == nullptr) { WHOLEMEMORY_ERROR("Open file %s for read failed.", file_names[i]); }
    /*|***read_file_begin_entry_off***|***entry_gap***|***cur_file_read_entry_count***|******|*/
    while (read_file_begin_entry_off < file_entry_count) {
      //$fseek by remain_entry_gap
      if (read_file_begin_entry_off + next_entry_gap >= file_entry_count) {
        next_entry_gap = (read_file_begin_entry_off + next_entry_gap) - file_entry_count;
        break;
      }
      size_t file_read_start_offset = next_entry_gap * entry_size;
      if (fseeko(fp, file_read_start_offset, SEEK_CUR) != 0) {
        WHOLEMEMORY_ERROR("File %s seek to %ld failed.", file_names[i], file_read_start_offset);
      }
      size_t cur_file_read_entry_count;
      if (read_file_begin_entry_off + next_entry_gap + next_continuous_entry_count >
          file_entry_count) {
        cur_file_read_entry_count = file_entry_count - read_file_begin_entry_off - next_entry_gap;
        total_read_entry += cur_file_read_entry_count;
        read_file_begin_entry_off = file_entry_count;
        next_continuous_entry_count -= cur_file_read_entry_count;
        next_entry_gap = 0;
      } else {
        cur_file_read_entry_count = next_continuous_entry_count;
        total_read_entry += cur_file_read_entry_count;
        read_file_begin_entry_off += cur_file_read_entry_count + next_entry_gap;
        next_continuous_entry_count = round_robin_size > local_entry_count - total_read_entry
                                        ? local_entry_count - total_read_entry
                                        : round_robin_size;
        next_entry_gap              = (wm_world_size - 1) * round_robin_size;
      }
      read_size_from_cur_file += cur_file_read_entry_count * entry_size;
      // read cur_file_read_entry_count of embeddings
      size_t cur_file_read_entry = cur_file_read_entry_count;
      while (cur_file_read_entry_count > 0) {
        size_t read_entry_count = std::min(cur_file_read_entry_count, buffer_entry_count);
        int ret                 = fread(file_read_buffer.data(), entry_size, read_entry_count, fp);
        if (ret != read_entry_count) {
          WHOLEMEMORY_ERROR(
            "File %s line %d: reading from file %s, read_entry_count=%ld, entry_size=%ld, "
            "returned %d, error=%s\n",
            __FILE__,
            __LINE__,
            file_names[i],
            read_entry_count,
            entry_size,
            ret,
            strerror(errno));
        }
        if (entry_size != memory_entry_stride) {
          WM_CUDA_CHECK(cudaMemcpy2D(local_write_ptr,
                                     memory_entry_stride,
                                     file_read_buffer.data(),
                                     entry_size,
                                     entry_size,
                                     read_entry_count,
                                     cudaMemcpyDefault));
        } else {
          WM_CUDA_CHECK(cudaMemcpy(local_write_ptr,
                                   file_read_buffer.data(),
                                   read_entry_count * entry_size,
                                   cudaMemcpyDefault));
        }
        local_write_ptr += read_entry_count * memory_entry_stride;
        cur_file_read_entry_count -= read_entry_count;
      }
      if (total_read_entry > local_entry_count) {
        WHOLEMEMORY_ERROR(
          "file read error from rank %d, should read %lu entries, infact %lu entries.",
          wm_rank,
          local_entry_count,
          total_read_entry);
        break;
      } else if (total_read_entry == local_entry_count) {
        break;
      }
    }
    fclose(fp);
    WHOLEMEMORY_INFO("Rank=%d done Reading %ld bytes from file %s size=%ld",
                     wm_rank,
                     read_size_from_cur_file,
                     file_names[i],
                     file_sizes[i]);
    if (total_read_entry == local_entry_count) break;
  }
  WHOLEMEMORY_INFO("Rank=%d done Reading %ld entries, infact read %ld entries",
                   wm_rank,
                   total_read_entry,
                   local_entry_count);
}

/*!
 * Read from file list to local memory of WholeMemory. File list are binary files, which are
 * considered to be concatenated together. All ranks in WholeMemory will read the files in parallel
 * and load each part into local memory of each rank.
 * @param local_ptr : Pointer to local memory of WholeMemory
 * @param local_size : Local memory size
 * @param local_offset : The offset of local memory in WholeMemory.
 * @param entry_size : The entry size of each data entry.
 * @param memory_entry_stride : The stride of each entry in WholeMemory
 * @param memory_offset : The start offset to place the read data. Should be in range [0,
 * memory_entry_stride)
 * @param file_count : Total file count of the file list
 * @param file_names : File names of the file list.
 * @param file_sizes : Sizes of each file.
 * @param suggested_buffer_size : Suggested buffer size to read.
 * @param wm_rank : WholeMemory rank.
 */
static void read_file_list_to_local_memory(char* local_ptr,
                                           size_t local_size,
                                           size_t local_offset,
                                           size_t entry_size,
                                           size_t memory_entry_stride,
                                           size_t memory_offset,
                                           int file_count,
                                           const char** file_names,
                                           const std::vector<size_t>& file_sizes,
                                           size_t suggested_buffer_size,
                                           int wm_rank)
{
  size_t buffer_size;
  size_t buffer_entry_count = 1;
  if (suggested_buffer_size < entry_size) {
    buffer_size = entry_size;
  } else {
    buffer_entry_count = suggested_buffer_size / entry_size;
    buffer_size        = buffer_entry_count * entry_size;
  }
  std::vector<char> file_read_buffer(buffer_size);

  size_t local_entry_memory_start_index = local_offset / memory_entry_stride;
  size_t local_entry_file_start_index =
    local_entry_memory_start_index - memory_offset / memory_entry_stride;
  size_t local_entry_count = local_size / memory_entry_stride;
  char* local_write_ptr    = local_ptr + memory_offset % memory_entry_stride;
  if (wm_rank == 0) {
    local_entry_count -= memory_offset / memory_entry_stride;
    local_write_ptr += (memory_offset / memory_entry_stride) * memory_entry_stride;
  }
  size_t local_entry_idx = 0;

  size_t file_entry_offset = 0;
  size_t total_read_bytes  = 0;
  for (int i = 0; i < file_count; i++) {
    size_t file_entry_count = file_sizes[i] / entry_size;
    // already outside reading window
    if (file_entry_offset >= local_entry_file_start_index + local_entry_count) break;
    // in reading window
    if (file_entry_offset + file_entry_count > local_entry_file_start_index) {
      size_t file_read_start_offset = 0;
      FILE* fp                      = fopen(file_names[i], "rb");
      if (fp == nullptr) { WHOLEMEMORY_ERROR("Open file %s for read failed.", file_names[i]); }
      // maybe in window end, remove possible tailing data that don't belong to current rank.
      size_t to_read_file_entry_count = std::min(
        file_entry_count, local_entry_file_start_index + local_entry_count - file_entry_offset);
      // if in window begin, remove possible data that belongs to previous rank and skip disk
      // data.
      if (file_entry_offset < local_entry_file_start_index) {
        size_t skip_entry_count = local_entry_file_start_index - file_entry_offset;

        file_read_start_offset = skip_entry_count * entry_size;

        if (fseeko(fp, file_read_start_offset, SEEK_SET) != 0) {
          WHOLEMEMORY_ERROR(
            "File %s seek to %ld failed.", file_names[i], skip_entry_count * entry_size);
        }
        to_read_file_entry_count -= skip_entry_count;
      }
      // now all data in file_entry_count need to be read.
      size_t bytes_to_read    = to_read_file_entry_count * entry_size;
      size_t left_entry_count = to_read_file_entry_count;
      while (left_entry_count > 0) {
        size_t read_entry_count = std::min(left_entry_count, buffer_entry_count);

        int ret = fread(file_read_buffer.data(), entry_size, read_entry_count, fp);
        if (ret != read_entry_count) {
          WHOLEMEMORY_ERROR(
            "File %s line %d: reading from file %s, read_entry_count=%ld, entry_size=%ld, "
            "returned %d, error=%s\n",
            __FILE__,
            __LINE__,
            file_names[i],
            read_entry_count,
            entry_size,
            ret,
            strerror(errno));
        }

        if (entry_size != memory_entry_stride) {
          WM_CUDA_CHECK(cudaMemcpy2D(local_write_ptr,
                                     memory_entry_stride,
                                     file_read_buffer.data(),
                                     entry_size,
                                     entry_size,
                                     read_entry_count,
                                     cudaMemcpyDefault));
        } else {
          WM_CUDA_CHECK(cudaMemcpy(local_write_ptr,
                                   file_read_buffer.data(),
                                   read_entry_count * entry_size,
                                   cudaMemcpyDefault));
        }
        local_write_ptr += read_entry_count * memory_entry_stride;

        left_entry_count -= read_entry_count;
      }
      fclose(fp);
      WHOLEMEMORY_INFO(
        "Rank=%d done Reading %ld bytes from file %s size=%ld, starting from offset=%ld.",
        wm_rank,
        bytes_to_read,
        file_names[i],
        file_sizes[i],
        file_read_start_offset);
      total_read_bytes += bytes_to_read;
    }
    file_entry_offset += file_entry_count;
  }
  WHOLEMEMORY_INFO(
    "Rank=%d done reading total %ld bytes from needed files.", wm_rank, total_read_bytes);
}

/*!
 * Read from file list to local memory of WholeMemory using DirectIO. Using DirectIO may have better
 * performance by bypassing system cache if it is bottleneck. File list are binary files, which are
 * considered to be concatenated together. All ranks in WholeMemory will read the files in parallel
 * and load each part into local memory of each rank.
 * @param local_ptr : Pointer to local memory of WholeMemory
 * @param local_size : Local memory size
 * @param local_offset : The offset of local memory in WholeMemory.
 * @param entry_size : The entry size of each data entry.
 * @param memory_entry_stride : The stride of each entry in WholeMemory
 * @param memory_offset : The start offset to place the read data. Should be in range [0,
 * memory_entry_stride)
 * @param file_count : Total file count of the file list
 * @param file_names : File names of the file list.
 * @param file_sizes : Sizes of each file.
 * @param suggested_buffer_size : Suggested buffer size to read.
 * @param wm_rank : WholeMemory rank.
 */
static void read_file_list_to_local_memory_directio(char* local_ptr,
                                                    size_t local_size,
                                                    size_t local_offset,
                                                    size_t entry_size,
                                                    size_t memory_entry_stride,
                                                    size_t memory_offset,
                                                    int file_count,
                                                    const char** file_names,
                                                    const std::vector<size_t>& file_sizes,
                                                    size_t suggested_buffer_size,
                                                    int wm_rank)
{
  if (memory_offset + entry_size > memory_entry_stride) {
    WHOLEMEMORY_FAIL_NOTHROW("Direct io mode only support reading all entries.");
  }
  size_t local_entry_start_index = local_offset / memory_entry_stride;
  size_t local_entry_count       = local_size / memory_entry_stride;
  char* local_write_ptr          = local_ptr + memory_offset % memory_entry_stride;

  static size_t kAlignSize = 16 * 1024 * 1024;
  suggested_buffer_size    = round_up_unsafe<size_t>(suggested_buffer_size, kAlignSize);

  char* block_buffer;
  WHOLEMEMORY_CHECK_NOTHROW(posix_memalign(reinterpret_cast<void**>(&block_buffer),
                                           kAlignSize,
                                           suggested_buffer_size) == 0);

  size_t file_entry_offset = 0;
  size_t read_entry_count  = 0;
  for (int i = 0; i < file_count; i++) {
    size_t file_entry_count = file_sizes[i] / entry_size;
    // already outside reading window
    if (file_entry_offset >= local_entry_start_index + local_entry_count) break;
    // reading window not reached
    if (file_entry_offset + file_entry_count <= local_entry_start_index) {
      file_entry_offset += file_entry_count;
      continue;
    }
    // in reading window
    auto block_size = StatFileBlockSize(file_names[i]);
    if (block_size == 0 || block_size == (size_t)-1 || kAlignSize % block_size != 0) {
      WHOLEMEMORY_FAIL_NOTHROW(
        "block_size=%ld for file %s, but alignment is %ld", block_size, file_names[i], kAlignSize);
    }
    size_t buffer_block_count = suggested_buffer_size / block_size;
    int fd                    = open(file_names[i], O_DIRECT | O_RDONLY);
    if (fd < 0) { WHOLEMEMORY_FAIL_NOTHROW("Open file %s with direct io failed.", file_names[i]); }

    // maybe in window end, remove possible tailing data that don't belong to current rank.
    size_t to_read_file_entry_count =
      std::min(file_entry_count, local_entry_start_index + local_entry_count - file_entry_offset);

    size_t file_read_end = to_read_file_entry_count * entry_size;
    // if in window begin, remove possible data that belongs to previous rank and skip disk
    // data.
    size_t file_read_start = 0;
    if (file_entry_offset < local_entry_start_index) {
      size_t skip_entry_count = local_entry_start_index - file_entry_offset;
      to_read_file_entry_count -= skip_entry_count;
      file_read_start = skip_entry_count * entry_size;
    }

    size_t file_block_read_offset = file_read_start / block_size * block_size;
    size_t skip_head_size         = file_read_start - file_block_read_offset;

    char* local_mem_write_entry_for_file = local_write_ptr + read_entry_count * memory_entry_stride;
    size_t first_mem_entry_offset        = 0;
    size_t useful_data_bytes_read        = 0;
    size_t physical_data_bytes_read      = 0;
    while (file_block_read_offset < file_read_end) {
      size_t left_size          = file_read_end - file_block_read_offset;
      size_t left_block_count   = div_rounding_up_unsafe(left_size, block_size);
      size_t read_block_count   = std::min(left_block_count, buffer_block_count);
      size_t physical_read_size = read_block_count * block_size;
      physical_data_bytes_read += physical_read_size;

      ssize_t pread_size = pread64(fd, block_buffer, physical_read_size, file_block_read_offset);
      if (pread_size != physical_read_size &&
          file_block_read_offset + pread_size != file_sizes[i]) {
        WHOLEMEMORY_FAIL_NOTHROW(
          "rank=%d, pread_size=%ld, physical_read_size=%ld, file_block_read_offset=%ld, "
          "file_sizes[i]=%ld, file=%s",
          wm_rank,
          pread_size,
          physical_read_size,
          file_block_read_offset,
          file_sizes[i],
          file_names[i]);
      }

      size_t drop_tail_size = 0;
      if (file_block_read_offset + physical_read_size > file_read_end) {
        drop_tail_size = file_block_read_offset + physical_read_size - file_read_end;
      }

      char* useful_data_ptr   = block_buffer + skip_head_size;
      size_t useful_data_size = physical_read_size - skip_head_size - drop_tail_size;

      useful_data_bytes_read += useful_data_size;

      if (first_mem_entry_offset != 0) {
        // process head
        size_t entry_left_size = entry_size - first_mem_entry_offset;
        WM_CUDA_CHECK_NO_THROW(cudaMemcpy(local_mem_write_entry_for_file + first_mem_entry_offset,
                                          useful_data_ptr,
                                          entry_left_size,
                                          cudaMemcpyDefault));
        local_mem_write_entry_for_file += memory_entry_stride;
        useful_data_ptr += entry_left_size;
        useful_data_size -= entry_left_size;
        entry_left_size = 0;
      }

      size_t full_entry_count = useful_data_size / entry_size;
      size_t full_entry_size  = full_entry_count * entry_size;

      if (full_entry_size > 0) {
        if (entry_size != memory_entry_stride) {
          WM_CUDA_CHECK(cudaMemcpy2D(local_mem_write_entry_for_file,
                                     memory_entry_stride,
                                     useful_data_ptr,
                                     entry_size,
                                     entry_size,
                                     full_entry_count,
                                     cudaMemcpyDefault));
        } else {
          WM_CUDA_CHECK(cudaMemcpy(
            local_mem_write_entry_for_file, useful_data_ptr, full_entry_size, cudaMemcpyDefault));
        }
        local_mem_write_entry_for_file += memory_entry_stride * full_entry_count;
        useful_data_ptr += full_entry_size;
        useful_data_size -= full_entry_size;
      }

      size_t tail_entry_size = useful_data_size % entry_size;
      if (tail_entry_size != 0) {
        // process tail
        WM_CUDA_CHECK_NO_THROW(cudaMemcpy(
          local_mem_write_entry_for_file, useful_data_ptr, tail_entry_size, cudaMemcpyDefault));
        first_mem_entry_offset = tail_entry_size;
      }

      file_block_read_offset += physical_read_size;
      skip_head_size = 0;
    }

    WHOLEMEMORY_INFO(
      "Rank=%d done Reading %ld useful bytes by reading %ld block bytes using DirectIO from file "
      "%s size=%ld.",
      wm_rank,
      useful_data_bytes_read,
      physical_data_bytes_read,
      file_names[i],
      file_sizes[i]);

    close(fd);
    file_entry_offset += file_entry_count;
    read_entry_count += to_read_file_entry_count;
  }
  free(block_buffer);
}

wholememory_error_code_t load_file_to_handle(wholememory_handle_t wholememory_handle,
                                             size_t memory_offset,
                                             size_t memory_entry_stride,
                                             size_t entry_size,
                                             const char** file_names,
                                             int file_count,
                                             int round_robin_size) noexcept
{
  if (entry_size <= 0 || memory_offset < 0 || memory_offset + entry_size > memory_entry_stride) {
    WHOLEMEMORY_ERROR("Invalid input, entry_size=%ld, memory_entry_stride=%ld, memory_offset=%ld",
                      entry_size,
                      memory_entry_stride,
                      memory_offset);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  size_t wm_data_granularity = wholememory_get_data_granularity(wholememory_handle);
  if (wm_data_granularity % memory_entry_stride != 0) {
    WHOLEMEMORY_ERROR("Invalid input, memory_entry_stride=%ld, but wm_data_granularity=%ld",
                      memory_entry_stride,
                      wm_data_granularity);
    return WHOLEMEMORY_INVALID_INPUT;
  }

  size_t wm_total_size = wholememory_get_total_size(wholememory_handle);
  size_t expected_file_size =
    get_handle_partial_size(wm_total_size, memory_offset, memory_entry_stride, entry_size);

  if (file_count < 0 || file_count >= 65536) {
    WHOLEMEMORY_ERROR("input file count=%d", file_count);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  std::vector<size_t> file_sizes(file_count, 0);

  size_t file_total_size = 0;

  for (int i = 0; i < file_count; i++) {
    if (file_names[i] == nullptr) {
      WHOLEMEMORY_ERROR("input file %d of %d is nullptr.", i, file_count);
      return WHOLEMEMORY_INVALID_INPUT;
    }
    if (!IsFileExist(file_names[i], R_OK)) {
      WHOLEMEMORY_ERROR(
        "input_file[%d] of %d (%s) cannot open for read.", i, file_count, file_names[i]);
      return WHOLEMEMORY_INVALID_INPUT;
    }
    file_sizes[i] = StatFileSize(file_names[i]);
    if (file_sizes[i] == static_cast<size_t>(-1)) {
      WHOLEMEMORY_ERROR(
        "input_file[%d] of %d (%s) stat size failed.", i, file_count, file_names[i]);
      return WHOLEMEMORY_INVALID_INPUT;
    }
    if (file_sizes[i] % entry_size != 0) {
      WHOLEMEMORY_ERROR("input_file[%d] of %d (%s) size=%ld, but entry_size=%ld failed.",
                        i,
                        file_count,
                        file_names[i],
                        file_sizes[i],
                        entry_size);
      return WHOLEMEMORY_INVALID_INPUT;
    }
    file_total_size += file_sizes[i];
  }

  if (file_total_size > expected_file_size) {
    WHOLEMEMORY_ERROR("all %d input file size is %ld, but expected %ld",
                      file_count,
                      file_total_size,
                      expected_file_size);
    return WHOLEMEMORY_INVALID_VALUE;
  }

  try {
    wholememory_comm_t wm_comm;
    WHOLEMEMORY_CHECK(wholememory_get_communicator(&wm_comm, wholememory_handle) ==
                      WHOLEMEMORY_SUCCESS);

    int wm_rank, wm_world_size;
    WHOLEMEMORY_CHECK(wholememory_communicator_get_rank(&wm_rank, wm_comm) == WHOLEMEMORY_SUCCESS);
    WHOLEMEMORY_CHECK(wholememory_communicator_get_size(&wm_world_size, wm_comm) ==
                      WHOLEMEMORY_SUCCESS);
    WM_COMM_CHECK_ALL_SAME(wm_comm, file_count);

    for (int i = 0; i < file_count; i++) {
      WM_COMM_CHECK_ALL_SAME(wm_comm, file_sizes[i]);
    }

    char* local_ptr = nullptr;
    size_t local_size, local_offset;

    WHOLEMEMORY_CHECK(wholememory_get_local_memory(
                        (void**)(&local_ptr), &local_size, &local_offset, wholememory_handle) ==
                      WHOLEMEMORY_SUCCESS);

    int suggested_buffer_size_mb    = 16;
    const char* buffer_size_env_var = std::getenv("WG_LOAD_BUFFER_SIZE_MB");
    if (buffer_size_env_var != nullptr) {
      try {
        suggested_buffer_size_mb = std::stoi(buffer_size_env_var);
      } catch (const std::invalid_argument& e) {
        suggested_buffer_size_mb = 16;
        WHOLEMEMORY_WARN(
          "Environment variable WG_LOAD_BUFFER_SIZE_MB value %s is not valid, using default %d",
          buffer_size_env_var,
          suggested_buffer_size_mb);
      }
      if (suggested_buffer_size_mb < 1) {
        suggested_buffer_size_mb = 16;
        WHOLEMEMORY_WARN(
          "Environment variable WG_LOAD_BUFFER_SIZE_MB value %s is not valid, using default %d",
          buffer_size_env_var,
          suggested_buffer_size_mb);
      }
    }
    size_t suggested_buffer_size = static_cast<size_t>(suggested_buffer_size_mb) * 1024 * 1024;

    const char* directio_env_var = std::getenv("WG_LOAD_USE_DIRECTIO");
    bool use_direct_io           = false;
    if (directio_env_var != nullptr && directio_env_var[0] == '1' && directio_env_var[1] == '\0') {
      use_direct_io = true;
    }
    if (round_robin_size != 0) {
      read_file_list_to_local_memory_roundrobin(local_ptr,
                                                local_size,
                                                local_offset,
                                                entry_size,
                                                memory_entry_stride,
                                                memory_offset,
                                                file_count,
                                                file_names,
                                                file_sizes,
                                                suggested_buffer_size,
                                                wm_rank,
                                                wm_world_size,
                                                round_robin_size);
    } else if (!use_direct_io) {
      read_file_list_to_local_memory(local_ptr,
                                     local_size,
                                     local_offset,
                                     entry_size,
                                     memory_entry_stride,
                                     memory_offset,
                                     file_count,
                                     file_names,
                                     file_sizes,
                                     suggested_buffer_size,
                                     wm_rank);
    } else {
      read_file_list_to_local_memory_directio(local_ptr,
                                              local_size,
                                              local_offset,
                                              entry_size,
                                              memory_entry_stride,
                                              memory_offset,
                                              file_count,
                                              file_names,
                                              file_sizes,
                                              suggested_buffer_size,
                                              wm_rank);
    }

    wm_comm->barrier();
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("Logic error: %s", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("CUDA error: %s", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (...) {
    WHOLEMEMORY_ERROR("Unknow error caught at file %s, line %d", __FILE__, __LINE__);
    return WHOLEMEMORY_UNKNOW_ERROR;
  }

  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t store_handle_to_file(wholememory_handle_t wholememory_handle,
                                              size_t memory_offset,
                                              size_t memory_entry_stride,
                                              size_t entry_size,
                                              const char* local_file_name) noexcept
{
  if (entry_size <= 0 || memory_offset < 0 || memory_offset + entry_size > memory_entry_stride) {
    WHOLEMEMORY_ERROR("Invalid input, entry_size=%ld, memory_entry_stride=%ld, memory_offset=%ld",
                      entry_size,
                      memory_entry_stride,
                      memory_offset);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  size_t wm_data_granularity = wholememory_get_data_granularity(wholememory_handle);
  if (wm_data_granularity % memory_entry_stride != 0) {
    WHOLEMEMORY_ERROR("Invalid input, memory_entry_stride=%ld, but wm_data_granularity=%ld",
                      memory_entry_stride,
                      wm_data_granularity);
    return WHOLEMEMORY_INVALID_INPUT;
  }

  try {
    wholememory_comm_t wm_comm;
    WHOLEMEMORY_CHECK(wholememory_get_communicator(&wm_comm, wholememory_handle) ==
                      WHOLEMEMORY_SUCCESS);

    int wm_rank;
    WHOLEMEMORY_CHECK(wholememory_communicator_get_rank(&wm_rank, wm_comm) == WHOLEMEMORY_SUCCESS);

    char* local_ptr = nullptr;
    size_t local_size, local_offset;

    wm_comm->barrier();

    WHOLEMEMORY_CHECK(wholememory_get_local_memory(
                        (void**)(&local_ptr), &local_size, &local_offset, wholememory_handle) ==
                      WHOLEMEMORY_SUCCESS);

    constexpr int kSuggestedBufferSize = 16 * 1024 * 1024;
    size_t buffer_size;
    size_t buffer_entry_count = 1;
    if (kSuggestedBufferSize < entry_size) {
      buffer_size = entry_size;
    } else {
      buffer_entry_count = kSuggestedBufferSize / entry_size;
      buffer_size        = buffer_entry_count * entry_size;
    }
    std::vector<char> file_write_buffer(buffer_size);

    size_t local_entry_count = local_size / memory_entry_stride;
    char* local_write_ptr    = local_ptr + memory_offset % memory_entry_stride;
    if (wm_rank == 0) {
      local_entry_count -= memory_offset / memory_entry_stride;
      local_write_ptr += (memory_offset / memory_entry_stride) * memory_entry_stride;
    }

    FILE* fp = fopen(local_file_name, "wb");
    if (fp == nullptr) {
      WHOLEMEMORY_ERROR("Rank=%d, open output file %s failed.\n", wm_rank, local_file_name);
    }

    size_t left_entry_count = local_entry_count;
    while (left_entry_count > 0) {
      size_t write_entry_count = std::min(left_entry_count, buffer_entry_count);
      if (entry_size != memory_entry_stride) {
        WM_CUDA_CHECK(cudaMemcpy2D(file_write_buffer.data(),
                                   entry_size,
                                   local_write_ptr,
                                   memory_entry_stride,
                                   entry_size,
                                   write_entry_count,
                                   cudaMemcpyDefault));
      } else {
        WM_CUDA_CHECK(cudaMemcpy(file_write_buffer.data(),
                                 local_write_ptr,
                                 write_entry_count * entry_size,
                                 cudaMemcpyDefault));
      }
      local_write_ptr += write_entry_count * memory_entry_stride;
      int ret = fwrite(file_write_buffer.data(), entry_size, write_entry_count, fp);

      if (ret != write_entry_count) {
        WHOLEMEMORY_ERROR(
          "File %s line %d: writing to file %s, write_entry_count=%ld, entry_size=%ld, "
          "returned %d, error=%s\n",
          __FILE__,
          __LINE__,
          local_file_name,
          write_entry_count,
          entry_size,
          ret,
          strerror(errno));
      }

      left_entry_count -= write_entry_count;
    }

    fclose(fp);

    WHOLEMEMORY_INFO("Rank=%d done writing to file %s.", wm_rank, local_file_name);

    wm_comm->barrier();

  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("Logic error: %s", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("CUDA error: %s", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (...) {
    WHOLEMEMORY_ERROR("Unknow error caught at file %s, line %d", __FILE__, __LINE__);
    return WHOLEMEMORY_UNKNOW_ERROR;
  }

  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory
