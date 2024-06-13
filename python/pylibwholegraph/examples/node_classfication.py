# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

import datetime
import os
import time
import argparse

import apex
import torch
from apex.parallel import DistributedDataParallel as DDP
import pylibwholegraph.torch as wgth


argparser = argparse.ArgumentParser()
wgth.add_distributed_launch_options(argparser)
wgth.add_training_options(argparser)
wgth.add_common_graph_options(argparser)
wgth.add_common_model_options(argparser)
wgth.add_common_sampler_options(argparser)
wgth.add_node_classfication_options(argparser)
wgth.add_dataloader_options(argparser)
argparser.add_argument(
    "--fp16_embedding", action="store_true", dest="fp16_mbedding", default=False, help="Whether to use fp16 embedding"
)
args = argparser.parse_args()


def valid_test(dataloader, model, name):
    total_correct = 0
    total_valid_sample = 0
    if wgth.get_rank() == 0:
        print("%s..." % (name,))
    for i, (idx, label) in enumerate(dataloader):
        label = torch.reshape(label, (-1,)).cuda()
        model.eval()
        logits = model(idx)
        pred = torch.argmax(logits, 1)
        correct = (pred == label).sum()
        total_correct += correct.cpu()
        total_valid_sample += label.shape[0]
    if wgth.get_rank() == 0:
        print(
            "[%s] [%s] accuracy=%5.2f%%"
            % (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                name,
                100.0 * total_correct / total_valid_sample,
            )
        )


def valid(valid_dataloader, model):
    valid_test(valid_dataloader, model, "VALID")


def test(test_dataset, model):
    test_dataloader = wgth.get_valid_test_dataloader(test_dataset, args.batchsize)
    valid_test(test_dataloader, model, "TEST")


def train(train_data, valid_data, model, optimizer, wm_optimizer, global_comm):
    if wgth.get_rank() == 0:
        print("start training...")
    train_dataloader = wgth.get_train_dataloader(
        train_data,
        args.batchsize,
        replica_id=wgth.get_rank(),
        num_replicas=wgth.get_world_size(),
        num_workers=args.dataloaderworkers,
    )
    valid_dataloader = wgth.get_valid_test_dataloader(valid_data, args.batchsize)
    valid(valid_dataloader, model)

    train_step = 0
    epoch = 0
    loss_fcn = torch.nn.CrossEntropyLoss()
    train_start_time = time.time()
    while epoch < args.epochs:
        for i, (idx, label) in enumerate(train_dataloader):
            label = torch.reshape(label, (-1,)).cuda()
            optimizer.zero_grad()
            model.train()
            logits = model(idx)
            loss = loss_fcn(logits, label)
            loss.backward()
            optimizer.step()
            if wm_optimizer is not None:
                wm_optimizer.step(args.lr * 0.1)
            if wgth.get_rank() == 0 and train_step % 100 == 0:
                print(
                    "[%s] [LOSS] step=%d, loss=%f"
                    % (
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        train_step,
                        loss.cpu().item(),
                    )
                )
            train_step = train_step + 1
        epoch = epoch + 1
    global_comm.barrier()
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    if wgth.get_rank() == 0:
        print(
            "[%s] [TRAIN_TIME] train time is %.2f seconds"
            % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), train_time)
        )
        print(
            "[EPOCH_TIME] %.2f seconds."
            % ((train_end_time - train_start_time) / args.epochs,)
        )
    valid(valid_dataloader, model)


def main_func():
    print(f"Rank={wgth.get_rank()}, local_rank={wgth.get_local_rank()}")
    global_comm, local_comm = wgth.init_torch_env_and_create_wm_comm(
        wgth.get_rank(),
        wgth.get_world_size(),
        wgth.get_local_rank(),
        wgth.get_local_size(),
        args.distributed_backend_type,
        args.log_level
    )

    if args.use_cpp_ext:
        wgth.compile_cpp_extension()

    train_ds, valid_ds, test_ds = wgth.create_node_claffication_datasets(
        args.pickle_data_path
    )

    graph_structure = wgth.GraphStructure()
    graph_structure_wholememory_type = "chunked"
    graph_structure_wholememory_location = "cuda"

    graph_comm = local_comm
    if global_comm.get_size() != local_comm.get_size() and global_comm.support_type_location("continuous", "cuda"):
        print("Using global communicator for graph structure.")
        graph_comm = global_comm
        graph_structure_wholememory_type = "continuous"
        graph_structure_wholememory_location = "cuda"
        if not args.use_global_embedding:
            args.use_global_embedding = True
            print("Changing to using global communicator for embedding...")
            if args.embedding_memory_type == "chunked":
                print("Changing to continuous wholememory for embedding...")
                args.embedding_memory_type = "continuous"

    csr_row_ptr_wm_tensor = wgth.create_wholememory_tensor_from_filelist(
        graph_comm,
        graph_structure_wholememory_type,
        graph_structure_wholememory_location,
        os.path.join(args.root_dir, "homograph_csr_row_ptr"),
        torch.int64,
    )
    csr_col_ind_wm_tensor = wgth.create_wholememory_tensor_from_filelist(
        graph_comm,
        graph_structure_wholememory_type,
        graph_structure_wholememory_location,
        os.path.join(args.root_dir, "homograph_csr_col_idx"),
        torch.int,
    )
    graph_structure.set_csr_graph(csr_row_ptr_wm_tensor, csr_col_ind_wm_tensor)

    feature_comm = global_comm if args.use_global_embedding else local_comm

    embedding_wholememory_type = args.embedding_memory_type
    embedding_wholememory_location = (
        "cpu" if args.cache_type != "none" or args.cache_ratio == 0.0 else "cuda"
    )
    if args.cache_ratio == 0.0:
        args.cache_type = "none"
    access_type = "readonly" if args.train_embedding is False else "readwrite"
    if wgth.get_rank() == 0:
        print(
            f"graph_structure: type={graph_structure_wholememory_type}, "
            f"location={graph_structure_wholememory_location}\n"
            f"embedding: type={embedding_wholememory_type}, location={embedding_wholememory_location}, "
            f"cache_type={args.cache_type}, cache_ratio={args.cache_ratio}, "
            f"trainable={args.train_embedding}, "
            f"distributed-backend-type={args.distributed_backend_type}, "
            f"use_global_embedding={args.use_global_embedding} "
        )
    cache_policy = wgth.create_builtin_cache_policy(
        args.cache_type,
        embedding_wholememory_type,
        embedding_wholememory_location,
        access_type,
        args.cache_ratio,
    )

    wm_optimizer = None
    embedding_dtype = torch.float32 if not args.fp16_mbedding else torch.float16
    if args.train_embedding is False:
        node_feat_wm_embedding = wgth.create_embedding_from_filelist(
            feature_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            os.path.join(args.root_dir, "node_feat.bin"),
            embedding_dtype,
            args.feat_dim,
            cache_policy=cache_policy,
            round_robin_size=args.round_robin_size,
        )
    else:
        node_feat_wm_embedding = wgth.create_embedding(
            feature_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            embedding_dtype,
            [graph_structure.node_count, args.feat_dim],
            cache_policy=cache_policy,
            random_init=True,
            round_robin_size=args.round_robin_size,
        )
        wm_optimizer = wgth.create_wholememory_optimizer(node_feat_wm_embedding, "adam", {})
    wgth.set_framework(args.framework)
    model = wgth.HomoGNNModel(graph_structure, node_feat_wm_embedding, args)
    model.cuda()
    model = DDP(model, delay_allreduce=True)
    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=args.lr)

    train(train_ds, valid_ds, model, optimizer, wm_optimizer, global_comm)
    test(test_ds, model)

    wgth.finalize()


if __name__ == "__main__":
    wgth.distributed_launch(args, main_func)
