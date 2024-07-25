import os, torch, random
import numpy as np
from models.Semi_SM_model import Semi_SM_model
import warnings
warnings.filterwarnings("ignore")
from utils.data_utils_mm import get_loader
from monai.transforms import Activations, AsDiscrete, Compose
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
import torch.nn.functional as F
from utils.utils import get_dice_score

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def validation(model, ValLoader, args):
    model.eval()
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS))  # 1st row for dice, 2nd row for count
    dice_CT_arr =[]
    dice_MRI_arr =[]
    with torch.no_grad():
        for index, batch in enumerate(tqdm(ValLoader)):
            CT_image, CT_seg, MRI_image, MRI_seg, label_id, name = batch["CT_image"], batch["CT_seg"], \
                batch["MRI_image"], batch["MRI_seg"], batch["label"], batch["name"]
            CT_image, CT_seg, MRI_image, MRI_seg = CT_image.cuda(args.rank), CT_seg.cuda(args.rank), MRI_image.cuda(
                args.rank), MRI_seg.cuda(args.rank)
            with autocast(enabled=True):
                CT_img_F_ds, MRI_img_F_ds, CT_seg_out, MRI_seg_out = model(CT_image, MRI_image)
            dice_CT = get_dice_score(CT_seg_out, CT_seg)
            dice_MRI = get_dice_score(MRI_seg_out, MRI_seg)
            dice_CT_arr.append(dice_CT)
            dice_MRI_arr.append(dice_MRI)
            name =name[0].split('/')[-1]
            torch.cuda.empty_cache()
            print("name:{}, dice_MRI:{:.4f}, dice_CT:{:.4f}".format(name, dice_CT, dice_MRI))
        avg_dice_CT = np.mean(dice_CT_arr)
        avg_dice_MRI = np.mean(dice_MRI_arr) 
        print("avg_dice_CT:{:.4f}, avg_dice_MRI:{:.4f}".format(avg_dice_CT, avg_dice_MRI))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='medical contest')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--img_size', default=96, type=int)
    parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
    parser.add_argument("--logdir", default="test_250", type=str, help="directory to save the tensorboard logs")
    parser.add_argument('--trained_weights', default=f"./checkpoint/model_20_perc_labeled.pt", type=str)
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--test_mode", default=1, type=int, help="node rank for distributed training")
    parser.add_argument('--backbone', default='Foundation_model', help='backbone [Foundation_model or VIT3D]')
    parser.add_argument("--workers", default=8, type=int, help="number of workers")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.3, type=float,
                        help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float,
                        help="RandShiftIntensityd aug probability")
    parser.add_argument("--amp", default=1, type=int, help="use amp for training")
    parser.add_argument("--save_checkpoint", default=1, type=int, help="save checkpoint during training")
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')
    args.NUM_CLASS = args.out_channels

    model = Semi_SM_model(img_size=args.img_size,
                    n_class=args.out_channels,
                    backbone=args.backbone
                    )
    model.load_state_dict(torch.load(args.trained_weights)['net'])
    model.to(device)

    train_loader, val_loader, test_loader = get_loader(args)

    validation(model, test_loader, args)

if __name__ == "__main__":
    setup_seed()
    main()