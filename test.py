import argparse
import os
import shutil
import nibabel as nib
import numpy as np
import torch
from utils.utils import dice, resample_3d
from utils.Semi_MM_data_utils import get_loader
from model.mm_Unet import UNet3D as Foundatiom_model

from model.SwinUNETR import SwinUNETR
from model.Semi_SM_model_MIA import Semi_SM_model
from utils.utils import dice, resample_3d, ORGAN_NAME

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./runs/saved_checkpoint/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="./dataset/mm_amos/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test_CMC_1", type=str, help="experiment name")
parser.add_argument("--json_list", default="cmc_data_list_example.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
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
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=1, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")

def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    ct_loader = get_loader(args,train_modality='CT')
    mri_loader = get_loader(args,train_modality='MRI')
    pretrained_dir = args.pretrained_dir
    model_name = "model.pt"
    num_class = args.out_channels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = Semi_SM_model(img_size=args.roi_x,
                              n_class=args.out_channels,
                              )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)
    with torch.no_grad():
        ct_spl_dice_all = []
        mri_spl_dice_all = []
        ct_Rkid_dice_all = []
        mri_Rkid_dice_all = []
        ct_Lkid_dice_all = []
        mri_Lkid_dice_all = []
        ct_Liver_dice_all = []
        mri_Liver_dice_all = []
        for i, (batch_ct, batch_mri) in enumerate(zip(ct_loader, mri_loader)):
            ct_data, ct_target, mri_data, mri_target = batch_ct["img_CT"].cuda(), batch_ct["mask_CT"].cuda(), batch_mri["img_MRI"].cuda(), batch_mri["mask_MRI"].cuda()
            ct_img_name, mri_img_name = batch_ct["img_CT_meta_dict"]["filename_or_obj"][0].split("/")[-1], batch_mri["img_MRI_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            ct_original_affine, mri_original_affine= batch_ct["img_CT_meta_dict"]["affine"][0].numpy(), batch_mri["img_MRI_meta_dict"]["affine"][0].numpy()
            _, _, ct_logits, mri_logits = model(ct_data, mri_data)
            pre_ct_outputs = torch.softmax(ct_logits, 1).cpu().numpy()
            pre_ct_outputs = np.argmax(pre_ct_outputs, axis=1).astype(np.uint8)[0]
            pre_mri_outputs = torch.softmax(mri_logits, 1).cpu().numpy()
            pre_mri_outputs = np.argmax(pre_mri_outputs, axis=1).astype(np.uint8)[0]
            _, _, h, w, d = ct_target.shape
            _, _, h1, w1, d1 = mri_target.shape
            ct_target_shape = (h, w, d)
            mri_target_shape = (h1, w1, d1)

            ori_ct_data, ori_mri_data= ct_data[0, 0].detach().cpu().numpy(), mri_data[0, 0].detach().cpu().numpy()
            ct_labels, mri_labels= ct_target.cpu().numpy()[0, 0, :, :, :], mri_target.cpu().numpy()[0, 0, :, :, :]
            pre_ct_outputs = resample_3d(pre_ct_outputs, ct_target_shape)
            ori_ct_data = resample_3d(ori_ct_data, ct_target_shape)
            pre_mri_outputs = resample_3d(pre_mri_outputs, mri_target_shape)
            ori_mri_data = resample_3d(ori_mri_data, mri_target_shape)
            # compute dice for each organ: liver, Spleen, Right Kidney and Left Kidney
            ct_Rkid_dice =0
            ct_Lkid_dice =0
            ct_spl_dice =0
            ct_Liver_dice =0
            mri_Rkid_dice =0
            mri_Lkid_dice =0
            mri_spl_dice =0
            mri_Liver_dice =0
            for i in range(1, num_class):
                organ_name = ORGAN_NAME[i - 1]
                if organ_name == 'Spleen':
                    ct_spl_dice = dice(pre_ct_outputs == i, ct_labels == i)
                    mri_spl_dice = dice(pre_mri_outputs == i, mri_labels == i)
                    ct_spl_dice_all.append(ct_spl_dice)
                    mri_spl_dice_all.append(mri_spl_dice)
                elif organ_name == 'Right Kidney':
                    ct_Rkid_dice = dice(pre_ct_outputs == i, ct_labels == i)
                    mri_Rkid_dice = dice(pre_mri_outputs == i, mri_labels == i)
                    ct_Rkid_dice_all.append(ct_Rkid_dice)
                    mri_Rkid_dice_all.append(mri_Rkid_dice)
                elif organ_name == 'Left Kidney':
                    ct_Lkid_dice = dice(pre_ct_outputs == i, ct_labels == i)
                    mri_Lkid_dice = dice(pre_mri_outputs == i, mri_labels == i)
                    ct_Lkid_dice_all.append(ct_Lkid_dice)
                    mri_Lkid_dice_all.append(mri_Lkid_dice)
                elif organ_name == 'Liver':
                    ct_Liver_dice = dice(pre_ct_outputs == i, ct_labels == i)
                    mri_Liver_dice = dice(pre_mri_outputs == i, mri_labels == i)
                    ct_Liver_dice_all.append(ct_Liver_dice)
                    mri_Liver_dice_all.append(mri_Liver_dice)
                elif i > 8:
                    break
            print("CT case name:{}, spleen dice: {}, R kidney dice: {}, L kidney dice: {}, Liver dice:{}, avg organ dice:{}".format(ct_img_name, ct_spl_dice, ct_Rkid_dice, ct_Lkid_dice, ct_Liver_dice, np.mean([ct_spl_dice, ct_Rkid_dice, ct_Lkid_dice, ct_Liver_dice])))
            print("MRI case name:{}, spleen dice: {}, R kidney dice: {}, L kidney dice: {}, Liver dice:{}, avg organ dice:{}".format(mri_img_name, mri_spl_dice, mri_Rkid_dice, mri_Lkid_dice, mri_Liver_dice, np.mean([mri_spl_dice, mri_Rkid_dice, mri_Lkid_dice, mri_Liver_dice])))
            # save the prediction result
            # nib.save(nib.Nifti1Image(pre_ct_outputs.astype(np.uint8), ct_original_affine), os.path.join(output_directory, ct_img_name.replace("CT","pre_CT")))
            # nib.save(nib.Nifti1Image(ori_ct_data, ct_original_affine), os.path.join(output_directory, ct_img_name.replace("CT","ori_CT")))
            # nib.save(nib.Nifti1Image(ct_labels, ct_original_affine), os.path.join(output_directory, ct_img_name.replace("CT","gt_CT")))
            # nib.save(nib.Nifti1Image(pre_mri_outputs.astype(np.uint8), mri_original_affine), os.path.join(output_directory, mri_img_name.replace("MRI","pre_MRI")))
            # nib.save(nib.Nifti1Image(ori_mri_data, mri_original_affine), os.path.join(output_directory, mri_img_name.replace("MRI","ori_MRI")))
            # nib.save(nib.Nifti1Image(mri_labels, mri_original_affine), os.path.join(output_directory, mri_img_name.replace("MRI","gt_MRI")))
        ct_spleen_dice_avg = np.mean(ct_spl_dice_all)
        mri_spleen_dice_avg = np.mean(mri_spl_dice_all)
        ct_R_kidney_dice_avg = np.mean(ct_Rkid_dice_all)
        mri_R_kidney_dice_avg = np.mean(mri_Rkid_dice_all)
        ct_L_kidney_dice_avg = np.mean(ct_Lkid_dice_all)
        mri_L_kidney_dice_avg = np.mean(mri_Lkid_dice_all)
        ct_Liver_dice_avg = np.mean(ct_Liver_dice_all)
        mri_Liver_dice_avg = np.mean(mri_Liver_dice_all)
        print("CT case: spleen avg dice: {}, R kidney avg dice: {}, L kidney avg dice : {}, Liver avg dice:{}".format(
                ct_spleen_dice_avg, ct_R_kidney_dice_avg, ct_L_kidney_dice_avg, ct_Liver_dice_avg))
        print("MRI case: spleen avg dice: {}, R kidney avg dice: {}, L kidney avg dice : {}, Liver avg dice:{}".format(
                mri_spleen_dice_avg, mri_R_kidney_dice_avg, mri_L_kidney_dice_avg, mri_Liver_dice_avg))
        ct_all_avg_dice=np.mean([ct_spleen_dice_avg,ct_R_kidney_dice_avg,ct_L_kidney_dice_avg,ct_Liver_dice_avg])
        mri_all_avg_dice=np.mean([mri_spleen_dice_avg,mri_R_kidney_dice_avg, mri_L_kidney_dice_avg, mri_Liver_dice_avg])
        print("CT Overall Mean Dice: {}, MRI Overall Mean Dice: {}".format(ct_all_avg_dice, mri_all_avg_dice))

if __name__ == "__main__":
    main()
