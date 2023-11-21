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
#include <cstddef>
#include <cstdlib>

#include "logger.hpp"
#include "nvshmem.h"
#include "nvshmemi_bootstrap.h"
#include "nvshmemx.h"
#include "wholememory/communicator.hpp"
#include "wholememory/initialize.hpp"
#include "wholememory/memory_handle.hpp"

#include <wholememory/wholememory.h>

// extern bool nvshmemi_is_nvshmem_bootstrapped;

static wholememory_comm_t bootstrap_comm = nullptr;

static int bootstrap_wholememory_barrier(struct bootstrap_handle* handle)
{
  bootstrap_comm->barrier();
  return WHOLEMEMORY_SUCCESS;
}

static int bootstrap_wholememory_allgather(const void* sendbuf,
                                           void* recvbuf,
                                           int length,
                                           struct bootstrap_handle* handle)

{
  bootstrap_comm->host_allgather(sendbuf, recvbuf, length, WHOLEMEMORY_DT_INT8);
  return WHOLEMEMORY_SUCCESS;
}

static int bootstrap_wholememory_alltoall(const void* sendbuf,
                                          void* recvbuf,
                                          int length,
                                          struct bootstrap_handle* handle)
{
  bootstrap_comm->host_alltoall(sendbuf, recvbuf, length, WHOLEMEMORY_DT_INT8);
  return WHOLEMEMORY_SUCCESS;
}

static void bootstrap_wholememory_global_exit(int status)
{
  try {
    bootstrap_comm->abort();
  } catch (const std::exception& e) {
    WHOLEMEMORY_ERROR("bootstrap_comm->abort() failed , error:%s\n", e.what());
    std::exit(1);
  }
}

static int bootstrap_wholememory_finalize(bootstrap_handle_t* handle)
{
  // do nothing
  return WHOLEMEMORY_SUCCESS;
}

int nvshmemi_bootstrap_plugin_init(void* wholememory_comm,
                                   bootstrap_handle_t* handle,
                                   const int abi_version)
{
  int status = 0, initialized = 0, finalized = 0;
  wholememory_comm_t src_comm;
  int bootstrap_version = NVSHMEMI_BOOTSTRAP_ABI_VERSION;

  WHOLEMEMORY_EXPECTS(
    nvshmemi_is_bootstrap_compatible(bootstrap_version, abi_version),
    "WholeMemory bootstrap version (%d) is not compatible with NVSHMEM version (%d)",
    bootstrap_version,
    abi_version);

  WHOLEMEMORY_EXPECTS((wholememory_comm != nullptr && wholememory_comm != NULL),
                      "WholeMemory bootstrap wholememory_comm should not == nullptr");
  src_comm = *((wholememory_comm_t*)wholememory_comm);

  bootstrap_comm      = src_comm;
  handle->pg_rank     = src_comm->world_rank;
  handle->pg_size     = src_comm->world_size;
  handle->allgather   = bootstrap_wholememory_allgather;
  handle->alltoall    = bootstrap_wholememory_alltoall;
  handle->barrier     = bootstrap_wholememory_barrier;
  handle->global_exit = bootstrap_wholememory_global_exit;
  handle->finalize    = bootstrap_wholememory_finalize;

  return status;
}
