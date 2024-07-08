import os, torch, random
import numpy as np
import pandas as pd
from monai import transforms
# from monai.data import DataLoader
from torch.utils.data import DataLoader
from trainer import trainer
from dataset.myDataset import myDataset
from models.mm_seg_model import multimodal_segmentation
# from medcam import medcam
from training import run_training

from optimizers.lr_scheduler import WarmupCosineSchedule,LinearWarmupCosineAnnealingLR
from oversampling.imbalanced import ImbalancedDatasetSampler
from utils.data_utils_mm import get_loader
from monai.losses import DiceCELoss,DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
import torch.nn as nn
from utils import loss
from segment_anything.build_sam3D import sam_model_registry3D
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(args.device)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model

def main():
    import argparse
    parser = argparse.ArgumentParser(description='medical contest')
    parser.add_argument('--max_epochs', default=5000, type=int)
    parser.add_argument('--val_every', default=30, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--resume', default=0, type=int, help='The path resume from checkpoint')
    # parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
    parser.add_argument("--logdir", default="test_main_250", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory")
    parser.add_argument("--train_data_dir", default="/train_test_data/", type=str, help="pretrained checkpoint directory")
    # parser.add_argument("--test_data_dir", default="/media/zxg/41dcb141-87ae-40f4-88ce-c566824e113b/home/zxg_big_file/RSNA_ADT/monai_process_data/test_npz/", type=str, help="pretrained checkpoint directory")
    parser.add_argument("--json_data_dir", default="dataset_list.json", type=str, help="pretrained checkpoint directory")
    # parser.add_argument("--test_json_data_dir", default="val_dataset.json", type=str, help="pretrained checkpoint directory")
    parser.add_argument('--log_dir', default="best_model", type=str)
    parser.add_argument('--model_name', default=f"foundation_model_sam_encoder", type=str)
    parser.add_argument('--pretrain', default=f"./pretrain_model/sam_vit_b_01ec64.pth", type=str)
    parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--test_mode", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--data_dir", default="./dataset/Training/", type=str, help="dataset directory")
    parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
    # add new  args
    parser.add_argument('--backbone', default='SAM-Med3D', help='backbone [SAM-Med3D or swinunetr or unet or dints or unetpp]')
    parser.add_argument("--workers", default=8, type=int, help="number of workers")
    parser.add_argument("--distributed", default=0, type=int, help="number of workers")

    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
    parser.add_argument("--use_normal_dataset", default=0, type=int, help="node rank for distributed training")
    parser.add_argument('--trans_encoding', default='word_embedding',
                        help='the type of encoding: rand_embedding or word_embedding')
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
    parser.add_argument("--RandScaleIntensityd_prob", default=0.3, type=float,
                        help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float,
                        help="RandShiftIntensityd aug probability")
    parser.add_argument("--squared_dice",default=1, type=int, help="squared_dice")
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=30, type=int, help="number of warmup epochs")
    parser.add_argument("--amp", default=1, type=int, help="use amp for training")
    parser.add_argument("--save_checkpoint", default=1, type=int, help="save checkpoint during training")
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    parser.add_argument('--consistency_type', type=str,
                        default="kl", help='consistency_type')
    parser.add_argument('--with_cons', type=str,
                        default="without_cons", help='with or without consistency')
    parser.add_argument('--consistency', type=float,
                        default=1.0, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=200.0, help='consistency_rampup')


    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    args.device = device
    torch.backends.cudnn.benchmark = True
    # CUDA_LAUNCH_BLOCKING = 1
    # TORCH_USE_CUDA_DSA = 1
    # Print All Config
    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    args.NUM_CLASS = args.out_channels
    # # model, optimizer and loss function

    # model = multimodal_segmentation(n_class=args.out_channels)
    model = build_model(args)
    model.load_params(torch.load(args.pretrain, map_location='cpu')['net'])
    model.to(device)

    # loader training and testing dataset
    loader = get_loader(args)

    if args.squared_dice:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr)
    else:
        dice_loss = DiceCELoss(to_onehot_y=args.NUM_CLASS, softmax=True)

    # dice_loss = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True, smooth_nr=args.smooth_nr,
    #                      smooth_dr=args.smooth_dr)
    # b_dice_loss = loss.BinaryDiceLoss().cuda()

    ce_loss = nn.CrossEntropyLoss().cuda()
    mse_loss = nn.MSELoss().cuda()
    post_label = AsDiscrete(to_onehot=args.NUM_CLASS, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.NUM_CLASS, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)

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


    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            for key in model_dict.keys():
                store_dict['.'.join(key.split('.')[1:])] = model_dict[key]
            model.load_state_dict(store_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])

        print('success resume from ', args.resume)

    dice = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=ce_loss,
        consistency_loss = dice_loss,
        contrastive_loss = mse_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return dice

    # trainer(model, train_loader, val_loader, optimizer, scheduler, dice_loss, args)

if __name__ == "__main__":
    setup_seed()
    main()