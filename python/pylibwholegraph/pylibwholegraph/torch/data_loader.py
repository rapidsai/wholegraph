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

import numpy as np
import torch
import pickle
from torch.utils.data import Dataset


class NodeClassificationDataset(Dataset):
    def __init__(self, raw_dataset):
        self.dataset = raw_dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def create_node_claffication_datasets(pickle_data_filename: str):
    with open(pickle_data_filename, "rb") as f:
        data_and_label = pickle.load(f)

    train_data = {
        "idx": data_and_label["train_idx"],
        "label": data_and_label["train_label"],
    }
    valid_data = {
        "idx": data_and_label["valid_idx"],
        "label": data_and_label["valid_label"],
    }
    test_data = {
        "idx": data_and_label["test_idx"],
        "label": data_and_label["test_label"],
    }
    train_dataset = list(
        list(zip(train_data["idx"], train_data["label"].astype(np.int64)))
    )
    valid_dataset = list(
        list(zip(valid_data["idx"], valid_data["label"].astype(np.int64)))
    )
    test_dataset = list(
        list(zip(test_data["idx"], test_data["label"].astype(np.int64)))
    )

    return (
        NodeClassificationDataset(train_dataset),
        NodeClassificationDataset(valid_dataset),
        NodeClassificationDataset(test_dataset),
    )


def get_train_dataloader(
    train_dataset,
    batch_size: int,
    *,
    replica_id: int = 0,
    num_replicas: int = 1,
    num_workers: int = 0
):
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_replicas,
        rank=replica_id,
        shuffle=True,
        drop_last=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else None,
        sampler=train_sampler,
    )
    return train_dataloader


def get_valid_test_dataloader(
    valid_test_dataset, batch_size: int, *, num_workers: int = 0
):
    valid_test_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_test_dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False
    )
    valid_test_dataloader = torch.utils.data.DataLoader(
        valid_test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=valid_test_sampler,
    )
    return valid_test_dataloader
