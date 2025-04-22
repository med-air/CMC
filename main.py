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

import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.Semi_MM_data_utils import get_loader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from model.Semi_SM_model import Semi_SM_model

from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from monai.losses import ContrastiveLoss
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="Semi_unet_monai_out_ori_96_cmc", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="./dataset/mm_amos/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="cmc_data_list_example.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.epoch.b4_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--save_checkpoint", default=1, type=int, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=500, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=30, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=0, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", default=1, type=int, help="use monai Dataset class")
parser.add_argument('--train_modality', default='MRI', type=str, choices=['CT', 'MRI', 'unlabeled'], help='CT or MRI' or 'unlabeled')
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
# parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_checkpoint", default=1, type=int, help="use gradient checkpointing to save memory")
# parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--use_ssl_pretrained", default=0, type=int, help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", default=1, type=int, help="use squared Dice")
parser.add_argument('--start_fusion_epoch', default=350, type=int)
parser.add_argument('--backbone', default='Semi_SM_model', choices=['Foundation_model','Semi_SM_model', 'SwinUNETR', 'VIT3D'], help='backbone [Foundation_model or SwinUNETR or VIT3D]')
parser.add_argument('--loss_opt', default='CSC', type=str, choices=['CSC', 'CAC', 'ALL'], help='select for loss')
parser.add_argument('--pretrain_dir', default=f"./pretrained_models/Foundation_model.pth", type=str)
parser.add_argument('--pretrain', default=1, type=int)

def CAC_loss(pred1, pred2, similarity='cosine'):
    """
    Compute CAC loss
    """
    if torch.sum(pred1) == 0 and torch.sum(pred2) == 0:
        return torch.tensor(1.0, device=pred1.device)
    smooth = 1e-6
    dim_len = len(pred1.size())
    if dim_len == 5:
       dim=(2,3,4)
    elif dim_len == 4:
       dim=(2,3)
    intersect = torch.sum(pred1 * pred2,dim=dim)
    y_sum = torch.sum(pred1 * pred1,dim=dim)
    z_sum = torch.sum(pred2 * pred2,dim=dim)
    dice_sim = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    dice_sim = dice_sim.mean()
    if torch.isnan(dice_sim):
        dice_sim = torch.tensor(1.0, device=dice_sim.device, requires_grad=True)
    return dice_sim

def CSC_loss(pred1,pred2):
    channel_losses = 0.0
    lens = pred1.shape[0]
    for c in range(pred1.shape[0]):
        pred1_output_channel = pred1[c, :, :, :]  # select the c_th channel of predi
        pred2_output_channel = pred2[c, :, :, :]  # select the c_th channel of pred2
        pred1_2d_flat = pred1_output_channel.reshape(-1, pred1_output_channel.shape[0])  # resize shape
        pred2_2d_flat = pred2_output_channel.reshape(-1, pred2_output_channel.shape[0])  # resize shape
        # compute the ContrastiveLoss from each channel
        cl_loss = ContrastiveLoss(batch_size=2, temperature=0.5)
        cl_value = cl_loss(pred1_2d_flat, pred2_2d_flat)
        channel_losses = channel_losses + cl_value
    mean_loss = channel_losses/ lens
    if torch.isnan(mean_loss):
        mean_loss = torch.tensor(1.0, device=mean_loss.device, requires_grad=True)
    return mean_loss

def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)
def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    ### load dataset
    ct_loader = get_loader(args,train_modality='CT')
    mri_loader = get_loader(args,train_modality='MRI')
    #### load unlabeled data
    ct_unlabeled_loader = get_loader(args,train_modality='ct_unlabeled')
    mri_unlabeled_loader = get_loader(args,train_modality='mri_unlabeled')

    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    pretrained_dir = args.pretrained_dir
    model =Semi_SM_model(img_size=args.roi_x,
                n_class=args.out_channels,
                )
    if args.resume_ckpt:
        model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))["state_dict"]
        model.load_state_dict(model_dict)
        print("Use pretrained weights")
    if args.pretrain is not None:
        model.load_encoder_params(torch.load(args.pretrain_dir, map_location='cpu'))
        model.load_decoder_params(torch.load(args.pretrain_dir, map_location='cpu'))
    if args.squared_dice:
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    unlabeled_model = model
    model.cuda(args.gpu)
    unlabeled_model.cuda(args.gpu)
    torch.cuda.empty_cache()

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        unlabeled_model=unlabeled_model,
        ct_train_loader=ct_loader[0],
        ct_val_loader=ct_loader[1],
        mri_train_loader=mri_loader[0],
        mri_val_loader=mri_loader[1],
        ct_unlabeled_loader=ct_unlabeled_loader,
        mri_unlabeled_loader=mri_unlabeled_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        CSC_loss=CSC_loss,
        CAC_loss=CAC_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
