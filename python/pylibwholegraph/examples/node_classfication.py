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

import datetime
import os
import time
from optparse import OptionParser

import apex
import torch
from apex.parallel import DistributedDataParallel as DDP

import pylibwholegraph.torch as wgth

parser = OptionParser()

wgth.add_distributed_launch_options(parser)
wgth.add_training_options(parser)
wgth.add_common_graph_options(parser)
wgth.add_common_model_options(parser)
wgth.add_common_sampler_options(parser)
wgth.add_node_classfication_options(parser)
wgth.add_dataloader_options(parser)
parser.add_option(
    "--fp16_embedding", action="store_true", dest="fp16_mbedding", default=False, help="Whether to use fp16 embedding"
)


(options, args) = parser.parse_args()


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
    test_dataloader = wgth.get_valid_test_dataloader(test_dataset, options.batchsize)
    valid_test(test_dataloader, model, "TEST")


def train(train_data, valid_data, model, optimizer, wm_optimizer, global_comm):
    if wgth.get_rank() == 0:
        print("start training...")
    train_dataloader = wgth.get_train_dataloader(
        train_data,
        options.batchsize,
        replica_id=wgth.get_rank(),
        num_replicas=wgth.get_world_size(),
        num_workers=options.dataloaderworkers,
    )
    valid_dataloader = wgth.get_valid_test_dataloader(valid_data, options.batchsize)
    valid(valid_dataloader, model)

    train_step = 0
    epoch = 0
    loss_fcn = torch.nn.CrossEntropyLoss()
    train_start_time = time.time()
    while epoch < options.epochs:
        for i, (idx, label) in enumerate(train_dataloader):
            label = torch.reshape(label, (-1,)).cuda()
            optimizer.zero_grad()
            model.train()
            logits = model(idx)
            loss = loss_fcn(logits, label)
            loss.backward()
            optimizer.step()
            if wm_optimizer is not None:
                wm_optimizer.step(options.lr * 0.1)
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
            % ((train_end_time - train_start_time) / options.epochs,)
        )
    valid(valid_dataloader, model)


def main_func():
    print(f"Rank={wgth.get_rank()}, local_rank={wgth.get_local_rank()}")
    global_comm, local_comm = wgth.init_torch_env_and_create_wm_comm(
        wgth.get_rank(),
        wgth.get_world_size(),
        wgth.get_local_rank(),
        wgth.get_local_size(),
    )
    if options.use_cpp_ext:
        wgth.compile_cpp_extension()

    train_ds, valid_ds, test_ds = wgth.create_node_claffication_datasets(
        options.pickle_data_path
    )

    graph_structure = wgth.GraphStructure()
    graph_structure_wholememory_type = "chunked"
    graph_structure_wholememory_location = "cuda"
    csr_row_ptr_wm_tensor = wgth.create_wholememory_tensor_from_filelist(
        local_comm,
        graph_structure_wholememory_type,
        graph_structure_wholememory_location,
        os.path.join(options.root_dir, "homograph_csr_row_ptr"),
        torch.int64,
    )
    csr_col_ind_wm_tensor = wgth.create_wholememory_tensor_from_filelist(
        local_comm,
        graph_structure_wholememory_type,
        graph_structure_wholememory_location,
        os.path.join(options.root_dir, "homograph_csr_col_idx"),
        torch.int,
    )
    graph_structure.set_csr_graph(csr_row_ptr_wm_tensor, csr_col_ind_wm_tensor)

    feature_comm = global_comm if options.use_global_embedding else local_comm
    wgth.comm_set_preferred_distributed_backend(feature_comm, options.distributed_backend_type)

    embedding_wholememory_type = options.embedding_memory_type
    embedding_wholememory_location = (
        "cpu" if options.cache_type != "none" or options.cache_ratio == 0.0 else "cuda"
    )
    if options.cache_ratio == 0.0:
        options.cache_type = "none"
    access_type = "readonly" if options.train_embedding is False else "readwrite"
    if wgth.get_rank() == 0:
        print(
            f"graph_structure: type={graph_structure_wholememory_type}, "
            f"location={graph_structure_wholememory_location}\n"
            f"embedding: type={embedding_wholememory_type}, location={embedding_wholememory_location}, "
            f"cache_type={options.cache_type}, cache_ratio={options.cache_ratio}, "
            f"trainable={options.train_embedding}, "
            f"distributed-backend-type={options.distributed_backend_type}, "
            f"use_global_embedding={options.use_global_embedding} "
        )
    cache_policy = wgth.create_builtin_cache_policy(
        options.cache_type,
        embedding_wholememory_type,
        embedding_wholememory_location,
        access_type,
        options.cache_ratio,
    )

    wm_optimizer = (
        None
        if options.train_embedding is False
        else wgth.create_wholememory_optimizer("adam", {})
    )

    embedding_dtype = torch.float32 if not options.fp16_mbedding else torch.float16

    if wm_optimizer is None:
        node_feat_wm_embedding = wgth.create_embedding_from_filelist(
            feature_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            os.path.join(options.root_dir, "node_feat.bin"),
            embedding_dtype,
            options.feat_dim,
            optimizer=wm_optimizer,
            cache_policy=cache_policy,
        )
    else:
        node_feat_wm_embedding = wgth.create_embedding(
            feature_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            embedding_dtype,
            [graph_structure.node_count, options.feat_dim],
            optimizer=wm_optimizer,
            cache_policy=cache_policy,
            random_init=True,
        )
    wgth.set_framework(options.framework)
    model = wgth.HomoGNNModel(graph_structure, node_feat_wm_embedding, options)
    model.cuda()
    model = DDP(model, delay_allreduce=True)
    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=options.lr)

    train(train_ds, valid_ds, model, optimizer, wm_optimizer, global_comm)
    test(test_ds, model)

    wgth.finalize()


if __name__ == "__main__":
    wgth.distributed_launch(options, main_func)
