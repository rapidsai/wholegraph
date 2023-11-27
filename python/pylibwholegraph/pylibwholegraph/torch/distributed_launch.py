# Copyright (c) 2019-2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser


class DistributedConfig(object):
    def __init__(self):
        super(DistributedConfig, self).__init__()
        self.rank = -1
        self.world_size = -1
        self.local_rank = -1
        self.local_size = -1
        self.master_addr = ""
        self.master_port = -1

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size

    def get_local_rank(self):
        return self.local_rank

    def get_local_size(self):
        return self.local_size

    def get_master_addr(self):
        return self.master_addr

    def get_master_port(self):
        return self.master_port


distributed_config = DistributedConfig()


def get_rank():
    global distributed_config
    return distributed_config.get_rank()


def get_world_size():
    global distributed_config
    return distributed_config.get_world_size()


def get_local_rank():
    global distributed_config
    return distributed_config.get_local_rank()


def get_master_addr():
    global distributed_config
    return distributed_config.get_master_addr()


def get_master_port():
    global distributed_config
    return distributed_config.get_master_port()


def get_local_size():
    global distributed_config
    return distributed_config.get_local_size()


def is_main_process():
    return get_rank() == 0


def add_distributed_launch_options(parser: ArgumentParser):
    parser.add_argument(
        "--launch-agent",
        dest="launch_agent",
        default="mpi",
        help="launch agent used, mpi, pytorch or spawn",
    )
    # command line flags
    parser.add_argument(
        "--rank", dest="rank", type=int, default=-1, help="command line flag for rank"
    )
    parser.add_argument(
        "--world-size",
        dest="world_size",
        type=int,
        default=-1,
        help="command line flag for world_size",
    )
    parser.add_argument(
        "--local-rank",
        dest="local_rank",
        type=int,
        default=-1,
        help="command line flag for local_rank",
    )
    parser.add_argument(
        "--local-size",
        dest="local_size",
        type=int,
        default=-1,
        help="command line flag for local_size",
    )
    parser.add_argument(
        "--master-addr",
        dest="master_addr",
        default="",
        help="command line flag for master_addr",
    )
    parser.add_argument(
        "--master-port",
        dest="master_port",
        type=int,
        default=-1,
        help="command line flag for master_port",
    )
    # environment variable names
    parser.add_argument(
        "--launch-env-name-world-rank",
        dest="launch_env_name_world_rank",
        default="RANK",
        help="environment variable name for world rank",
    )
    parser.add_argument(
        "--launch-env-name-world-size",
        dest="launch_env_name_world_size",
        default="WORLD_SIZE",
        help="environment variable name for world size",
    )
    parser.add_argument(
        "--launch-env-name-local-rank",
        dest="launch_env_name_local_rank",
        default="LOCAL_RANK",
        help="environment variable name for local rank",
    )
    parser.add_argument(
        "--launch-env-name-local-size",
        dest="launch_env_name_local_size",
        default="LOCAL_WORLD_SIZE",
        help="environment variable name for local size",
    )
    parser.add_argument(
        "--launch-env-name-master-addr",
        dest="launch_env_name_master_addr",
        default="MASTER_ADDR",
        help="environment variable name for master_addr",
    )
    parser.add_argument(
        "--launch-env-name-master-port",
        dest="launch_env_name_master_port",
        default="MASTER_PORT",
        help="environment variable name for master_port",
    )
    return


def get_value_from_env(env_name, fill_default=None):
    if env_name not in os.environ:
        if fill_default is not None:
            return fill_default
        else:
            raise ValueError(
                "both command line flag and environment %s not exist." % (env_name,)
            )
    else:
        return os.environ[env_name]


def get_value_from_option_and_env(
    option_value, env_name, empty_value, fill_default=None
):
    if option_value == empty_value:
        return get_value_from_env(env_name, fill_default)
    else:
        return option_value


def distributed_launch_mpi(args, main_func):
    from mpi4py import MPI

    mpi_communicator = MPI.COMM_WORLD
    shared_mpi_communicator = mpi_communicator.Split_type(MPI.COMM_TYPE_SHARED)

    global distributed_config
    distributed_config.rank = mpi_communicator.Get_rank()
    distributed_config.world_size = mpi_communicator.Get_size()
    distributed_config.local_rank = shared_mpi_communicator.Get_rank()
    distributed_config.local_size = shared_mpi_communicator.Get_size()
    distributed_config.master_addr = get_value_from_option_and_env(
        args.master_addr, args.launch_env_name_master_addr, "", "localhost"
    )
    distributed_config.master_port = int(
        get_value_from_option_and_env(
            args.master_port, args.launch_env_name_master_port, -1, 12335
        )
    )

    os.environ["RANK"] = str(distributed_config.rank)
    os.environ["WORLD_SIZE"] = str(distributed_config.world_size)
    os.environ["MASTER_ADDR"] = distributed_config.master_addr
    os.environ["MASTER_PORT"] = str(distributed_config.master_port)

    main_func()


def distributed_launch_pytorch(
    args,
    main_func,
):
    global distributed_config
    distributed_config.rank = int(
        get_value_from_env(args.launch_env_name_world_rank)
    )
    distributed_config.world_size = int(
        get_value_from_env(args.launch_env_name_world_size)
    )
    distributed_config.local_rank = int(
        get_value_from_option_and_env(
            args.local_rank, args.launch_env_name_local_rank, -1
        )
    )
    assert distributed_config.local_rank >= 0
    distributed_config.local_size = int(
        get_value_from_option_and_env(
            args.local_size, args.launch_env_name_local_size, -1
        )
    )
    assert distributed_config.local_size > 0
    distributed_config.master_addr = get_value_from_env(
        args.launch_env_name_master_addr
    )
    distributed_config.master_port = int(
        get_value_from_env(args.launch_env_name_master_port)
    )

    main_func()


def main_spawn_routine(local_rank, main_func, distributed_config_input):
    global distributed_config
    distributed_config = distributed_config_input
    node_rank = distributed_config.rank
    node_size = distributed_config.world_size

    distributed_config.rank = (
        node_rank * distributed_config.get_local_size() + local_rank
    )
    distributed_config.world_size = node_size * distributed_config.get_local_size()
    distributed_config.local_rank = local_rank

    os.environ["RANK"] = str(distributed_config.rank)
    os.environ["WORLD_SIZE"] = str(distributed_config.world_size)
    os.environ["MASTER_ADDR"] = distributed_config.master_addr
    os.environ["MASTER_PORT"] = str(distributed_config.master_port)

    main_func()


def distributed_launch_spawn(args, main_func):
    global distributed_config
    distributed_config.rank = int(
        get_value_from_option_and_env(
            args.rank, args.launch_env_name_world_rank, -1, 0
        )
    )
    distributed_config.world_size = int(
        get_value_from_option_and_env(
            args.world_size, args.launch_env_name_world_size, -1, 1
        )
    )
    distributed_config.local_rank = 0
    distributed_config.local_size = int(
        get_value_from_option_and_env(
            args.local_size, args.launch_env_name_local_size, -1, 1
        )
    )
    distributed_config.master_addr = get_value_from_option_and_env(
        args.master_addr, args.launch_env_name_master_addr, "", "localhost"
    )
    distributed_config.master_port = int(
        get_value_from_option_and_env(
            args.master_port, args.launch_env_name_master_port, -1, 12335
        )
    )

    import torch.multiprocessing as mp

    if distributed_config.local_size > 1:
        mp.spawn(
            main_spawn_routine,
            nprocs=distributed_config.local_size,
            args=(main_func, distributed_config),
        )
    else:
        main_spawn_routine(0, main_func, distributed_config)


def distributed_launch(args, main_func):
    assert (
        args.launch_agent == "mpi"
        or args.launch_agent == "pytorch"
        or args.launch_agent == "spawn"
    )
    if args.launch_agent == "mpi":
        # use MPI to launch multiprocess
        # when using MPI, command is like:
        # mpirun python [train_script.py]
        distributed_launch_mpi(args, main_func)
    elif args.launch_agent == "pytorch":
        # use pytorch DDP to launch multiprocess
        # when using pytorch DDP, assume two nodes with 8 GPU each, command is like:
        # on node1: python -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=node1
        #           --master_port=12335 [train_script.py] --launch_agent=pytorch
        # on node2: python -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=node1
        #           --master_port=12335 [train_script.py] --launch_agent=pytorch
        distributed_launch_pytorch(args, main_func)
    else:
        # cluster scheduler
        # when using spawn to create multiprocess for each node, assume two nodes with 8 GPU each, command is like:
        # on node1: python [train_script.py] --launch_agent=spawn --master_addr=node1 --master_port=12335
        #           --local_size=8 --rank=0 --world_size=2
        # on node2: python [train_script.py] --launch_agent=spawn --master_addr=node1 --master_port=12335
        #           --local_size=8 --rank=1 --world_size=2
        distributed_launch_spawn(args, main_func)
