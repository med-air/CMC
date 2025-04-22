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

    ### training ############
    file_root = "./dataset/dataset_amos/96_train_npz"
    labels_df = pd.read_csv('./dataset/dataset_amos/train_list_20_perc.csv',
                            index_col="Case_ID")
    train_images = []
    train_labels = []
    data_list = os.listdir(file_root)
    for index in data_list:
        train_images.append(os.path.join(file_root, index))
        name = index.split('.')[0]
        train_labels.append(labels_df.loc[name].values)
    train_labels = np.array(train_labels, dtype=float)
    train_labels = torch.FloatTensor(train_labels)

    ### validation ############
    val_file_root = "./dataset/dataset_amos/96_val_npz"
    val_labels_df = pd.read_csv('./dataset/dataset_amos/val_list.csv',
                                index_col="Case_ID")
    val_images = []
    val_labels = []
    val_data_list = os.listdir(val_file_root)
    for index in val_data_list:
        val_images.append(os.path.join(val_file_root, index))
        name = index.split('.')[0]
        val_labels.append(val_labels_df.loc[name].values)
    val_labels = np.array(val_labels, dtype=float)
    val_labels = torch.FloatTensor(val_labels)

    ### testing ############
    test_file_root = "./dataset/dataset_amos/96_test_npz_zxg"
    test_labels_df = pd.read_csv('./dataset/dataset_amos/test_list.csv',
        index_col="Case_ID")
    test_images = []
    test_labels = []
    test_data_list = os.listdir(test_file_root)
    for index in test_data_list:
        test_images.append(os.path.join(test_file_root, index))
        name = index.split('.')[0]
        test_labels.append(test_labels_df.loc[name].values)
    test_labels = np.array(test_labels, dtype=float)
    test_labels = torch.FloatTensor(test_labels)

    train_img_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            transforms.Orientation(axcodes="RAS"),
            transforms.RandFlip(prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlip(prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlip( prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90(prob=args.RandRotate90d_prob, max_k=3, spatial_axes=(1, 2)),
            transforms.RandScaleIntensity(factors=0.15, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensity(offsets=0.15, prob=args.RandShiftIntensityd_prob),
        ]
    )
    train_seg_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            transforms.RandFlip(prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlip(prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlip(prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90( prob=args.RandRotate90d_prob, max_k=3, spatial_axes=(1, 2)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
        ]
    )

    test_ds = myDataset(npz_files=test_images, labels=test_labels, img_transforms=test_transform,
                       seg_transforms=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    train_ds = myDataset(npz_files=train_images, labels=train_labels, img_transforms=train_img_transform,
                         seg_transforms=train_seg_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_ds = myDataset(npz_files=val_images, labels=val_labels, img_transforms=val_transform,
                       seg_transforms=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)


    return train_loader, val_loader, test_loader
