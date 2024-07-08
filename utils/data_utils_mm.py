# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from torch.utils.data import DataLoader

import numpy as np
import torch
from dataset.myDataset_mm import myDataset

from monai import data, transforms
from monai.data import load_decathlon_datalist

import pandas as pd

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):

    # data_dir = args.data_dir
    file_root = "/media/zxg/尼西/dataset/multi-modal-dataset/amos22/multi_modal_data/train_npz"
    labels_df = pd.read_csv('/media/zxg/尼西/dataset/multi-modal-dataset/amos22/multi_modal_data/train_list.csv',
                            index_col="Case_ID")
    train_images = []
    train_labels = []
    data_list = os.listdir(file_root)
    for index in data_list:
        train_images.append(os.path.join(file_root, index))
        name = index.split('.')[0]
        name = int(name)
        train_labels.append(labels_df.loc[name].values)
    train_labels = np.array(train_labels, dtype=float)
    train_labels = torch.FloatTensor(train_labels)

    val_file_root = "/media/zxg/尼西/dataset/multi-modal-dataset/amos22/multi_modal_data/val_npz"
    val_labels_df = pd.read_csv('/media/zxg/尼西/dataset/multi-modal-dataset/amos22/multi_modal_data/val_list.csv',
        index_col="Case_ID")
    val_images = []
    val_labels = []
    val_data_list = os.listdir(val_file_root)
    for index in val_data_list:
        val_images.append(os.path.join(val_file_root, index))
        name = index.split('.')[0]
        name = int(name)
        val_labels.append(val_labels_df.loc[name].values)
    val_labels = np.array(val_labels, dtype=float)
    val_labels = torch.FloatTensor(val_labels)

    train_img_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"]),
            # transforms.Resized(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"], spatial_size=(128, 128, 128),
            #                    mode="nearest"),
            # transforms.EnsureChannelFirstd(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"]),
            transforms.EnsureChannelFirst(channel_dim="no_channel"),

            # transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientation(axcodes="RAS"),
            # transforms.Spacing(pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
            # transforms.ScaleIntensityRanged(
            #     keys=["CT_image","MRI_image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            # transforms.CropForegroundd(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"], source_key="CT_label"),
            # transforms.RandCropByPosNegLabeld(
            #     keys=["CT_image", "CT_label", "MRI_image", "MRI_label"],
            #     label_key="CT_label",
            #     spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            #     pos=1,
            #     neg=1,
            #     num_samples=4,
            #     image_key="CT_image",
            #     image_threshold=0,
            # ),
            transforms.RandFlip(prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlip(prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlip( prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90(prob=args.RandRotate90d_prob, max_k=3, spatial_axes=(1, 2)),
            transforms.RandScaleIntensity(factors=0.15, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensity(offsets=0.15, prob=args.RandShiftIntensityd_prob),
            # transforms.ToTensord(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"]),
        ]
    )
    train_seg_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"]),
            transforms.EnsureChannelFirst(channel_dim="no_channel"),

            transforms.RandFlip(prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlip(prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlip(prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90( prob=args.RandRotate90d_prob, max_k=3, spatial_axes=(1, 2)),
            # transforms.ToTensord(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"]),
            # transforms.AddChanneld(keys=["image", "label"]),
            # transforms.Resized(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"], spatial_size=(128, 128, 128),
            #                    mode="nearest"),
            # transforms.EnsureChannelFirstd(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"]),
            transforms.EnsureChannelFirst(channel_dim="no_channel"),

            # transforms.Orientationd(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"], axcodes="RAS"),

            # transforms.Spacingd(
            #     keys=["CT_image", "CT_label", "MRI_image", "MRI_label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            # ),
            # transforms.ScaleIntensityRanged(
            #     keys=["CT_image","MRI_image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            # transforms.CropForegroundd(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"], source_key="CT_label"),
            # transforms.RandCropByPosNegLabeld(
            #     keys=["CT_image", "CT_label", "MRI_image", "MRI_label"],
            #     label_key="CT_label",
            #     spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            #     pos=1,
            #     neg=1,
            #     num_samples=4,
            #     image_key="CT_image",
            #     image_threshold=0,
            # ),
            # transforms.ToTensord(keys=["CT_image", "CT_label", "MRI_image", "MRI_label"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["image", "label"]),
            # transforms.AddChanneld(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        test_ds = myDataset(npz_files=val_images, labels=val_labels, img_transforms=test_transform,
                           seg_transforms=test_transform)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
        loader = test_loader
    else:
        train_ds = myDataset(npz_files=train_images, labels=train_labels, img_transforms=train_img_transform,
                             seg_transforms=train_seg_transform)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

        val_ds = myDataset(npz_files=val_images, labels=val_labels, img_transforms=val_transform,
                           seg_transforms=val_transform)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        loader = [train_loader, val_loader]

    return loader
