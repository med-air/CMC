import os, torch, random
import numpy as np
from models.Semi_SM_model import Semi_SM_model
from trainer import run_training
from segment_anything.build_sam3D import sam_model_registry3D
import warnings
warnings.filterwarnings("ignore")
from optimizers.lr_scheduler import WarmupCosineSchedule,LinearWarmupCosineAnnealingLR
from utils.data_utils_mm import get_loader
from monai.losses import DiceCELoss,SSIMLoss,ContrastiveLoss,DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from lightning import Fabric
from torch.backends import cudnn
import torch.distributed as dist
from monai.metrics import DiceMetric

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup(rank, world_size,args):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def main():
    import argparse
    parser = argparse.ArgumentParser(description='medical contest')
    parser.add_argument('--max_epochs', default=500, type=int)
    parser.add_argument('--val_every', default=30, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--model_type', type=str, default='vit_b_ori')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--img_size', default=96, type=int)
    parser.add_argument('--resume', default=0, type=int, help='The path resume from checkpoint')
    parser.add_argument("--optim_lr", default=1e-3, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--checkpoint", default="./test_250/2_epoch/model_final.pt", type=str, help="start training from saved checkpoint")
    parser.add_argument("--logdir", default="checkpoint/test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument('--pretrain', default=f"./pretrain_model/sam_vit_b_01ec64.pth", type=str)
    parser.add_argument('--de_pretrain', default=f"./pretrain_model/unet.pth", type=str)
    parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--test_mode", default=0, type=int, help="node rank for distributed training")
    parser.add_argument('--backbone', default='Foundation_model', help='backbone [Foundation_model or VIT3D]')
    parser.add_argument("--workers", default=8, type=int, help="number of workers")
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument('--port', type=int, default=12361)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1])
    parser.add_argument("--local_rank", type=int,default=1)
    parser.add_argument('--multi_gpu', action='store_true', default=True)
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
    parser.add_argument("--squared_dice",default=1, type=int, help="squared_dice")
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=150, type=int, help="number of warmup epochs")
    parser.add_argument("--amp", default=1, type=int, help="use amp for training")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.3, type=float,
                        help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float,
                        help="RandShiftIntensityd aug probability")
    parser.add_argument('--consistency_type', type=str,
                        default="kl", help='consistency_type')
    parser.add_argument('--with_cons', type=str,
                        default="without_cons", help='with or without consistency')
    parser.add_argument('--consistency', type=float,
                        default=1.0, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=500.0, help='consistency_rampup')
    parser.add_argument('--fusion_start_epoch', default=450, type=int)

    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    init_seeds(2023 + args.rank)

    def build_model(args):
        sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
        if args.multi_gpu:
            sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
        return sam_model

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    args.device = device
    if args.multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
    torch.backends.cudnn.benchmark = True
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')
    args.NUM_CLASS = args.out_channels

    model = Semi_SM_model(img_size=args.img_size,
                    n_class=args.out_channels,
                    backbone=args.backbone
                    )
    model.to(device)

    #Load pre-trained weights
    if args.pretrain is not None:
        model.load_encoder_params(torch.load(args.pretrain, map_location='cpu'))
        model.load_decoder_params(torch.load(args.de_pretrain, map_location='cpu')['net'])
    if args.dist and args.multi_gpu:
        args.nodes = 1
        args.ngpus_per_node = len(args.gpu_ids)
        world_size = args.nodes * args.ngpus_per_node
        rank = args.local_rank
        setup(rank, world_size,args)
        model = DDP(model, device_ids=[args.device])

    # loader training, validation and testing dataset
    loader = get_loader(args)
    
    if args.squared_dice:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr)
    else:
        dice_loss = DiceCELoss(to_onehot_y=args.NUM_CLASS, softmax=True)
    mse_loss = nn.MSELoss().cuda()
    # CSC_loss = similarity_loss_3D(img,gt)
    CSC_loss = SSIMLoss(spatial_dims=3,reduction='mean')
    diceCELoss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0
    if args.checkpoint is not None and args.resume==1:
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
        checkpoint = torch.load(args.checkpoint)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('success resume from ', args.resume)

    dice = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=diceCELoss,
        consistency_loss = mse_loss,
        contrastive_loss = mse_loss,
        dice_metric=dice_metric,
        args=args,
        model_inferer=model,
        scheduler=scheduler,
        start_epoch=start_epoch,
    )
    return dice

if __name__ == "__main__":
    setup_seed()
    main()