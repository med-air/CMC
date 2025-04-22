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

import numpy as np
import scipy.ndimage as ndimage
import torch

'''
AMOS dataset label:
"labels": {
    "00": "background",
    "01": "Spleen",
    "02": "Right Kidney",
    "03": "Left Kidney",
    "04": "Gall Bladder",
    "05": "Esophagus",
    "06": "Liver",
    "07": "Stomach",
    "08": "Aorta",
    "09": "Inferior Vena Cava",
    "10": "Pancreas",
    "11": "Right Adrenal Gland",
    "12": "Left Adrenal Gland",
    "13": "Duodenum",
    "14": "Bladder",
    "15": "Prostate"
}
'''

ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus',
                'Liver', 'Stomach', 'Aorta', 'Inferior Vena Cava', 'Pancreas',
                'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum']

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    if isinstance(mask_gt, np.ndarray):
        mask_gt = mask_gt.astype(bool)
        mask_pred = mask_pred.astype(bool)
    elif isinstance(mask_gt, torch.Tensor):
        mask_gt = mask_gt.bool()
        mask_pred = mask_pred.bool()
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def dice_score(preds, labels, spe_sen=False):  # on GPU
    ### preds: w,h,d; label: w,h,d
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = torch.tensor(preds)  # 转换为 Tensor
    labels = torch.tensor(labels)  # 转换为 Tensor
    preds = torch.where(preds > 0.5, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))
    fn = torch.sum(torch.mul(predict!=1, target))
    fp = torch.sum(torch.mul(predict, target!=1))
    tn = torch.sum(torch.mul(predict!=1, target!=1))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(fp + tn)


    # print(dice, recall, precision)
    if spe_sen:
        # return dice, recall, precision, specificity
        return dice
    else:
        # return dice, recall, precision
        return dice



def get_dice_score(prev_masks, gt3D): #refer to SAM-Med3D
    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = (mask_pred > mask_threshold)
        mask_gt = (mask_gt > 0)

        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2 * volume_intersect / volume_sum

    pred_masks = (prev_masks >= 0.5)
    true_masks = (gt3D > 0)
    dice_list = []
    for i in range(true_masks.shape[0]):
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))

    # 检查是否有有效的样本
    if len(dice_list) == 0:
        print("Warning: dice_list is empty. Returning default value 0.0")
        result = 0.0
    else:
        # result = (sum(dice_list) / len(dice_list)).item()
        result = (sum(dice_list) / len(dice_list))
    return result

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out
