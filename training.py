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

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from monai.data import decollate_batch
from scipy.spatial.distance import cdist
def my_sup_loss(loss_func1, loss_func2, loss_func3, CT_seg_out, MRI_seg_out, CT_seg, MRI_seg, CT_img_F_ds, MRI_img_F_ds, label_id):
    sup_losses = 0.0
    cac_losses = 0.0
    label_id_cnt = 0
    unlabel_id_cnt = 0
    csc_losses = 0.0
    tag = 0
    flag = 0
    for i in range(len(label_id)):
        if label_id[i] == 1:
            tag = 0
            flag = 1
            label_id_cnt = label_id_cnt + 1
            CT_sup_loss = loss_func1(CT_seg_out, CT_seg)
            MRI_sup_loss = loss_func1(MRI_seg_out, MRI_seg)
            csc_loss = my_csc_loss(loss_func2, CT_img_F_ds[i], MRI_img_F_ds[i])
            # csc_loss = loss_func2(CT_img_F_ds, MRI_img_F_ds)
            sup_loss = (CT_sup_loss + MRI_sup_loss) / 2
            sup_losses = sup_losses + sup_loss
            csc_losses = csc_losses + csc_loss
        else:
            tag = 1
            flag = 0
            unlabel_id_cnt = unlabel_id_cnt + 1
            cac_loss = loss_func3(CT_seg_out, MRI_seg_out)
            cac_losses = cac_losses + cac_loss
    sup_losses = sup_losses / (label_id_cnt + tag)
    csc_losses = csc_losses / (label_id_cnt + tag)
    cac_losses = cac_losses / (unlabel_id_cnt + flag)
    return sup_losses, csc_losses, cac_losses

def my_csc_loss(loss_fun,CT_out,MRI_out):
    channel_losses = 0.0
    lens = CT_out.shape[0]
    for c in range(CT_out.shape[0]):
        # CT_output_channel = CT_out[c, :, :, :].detach().cpu().numpy()  # 选择第 c 个通道,并增加通道维度
        CT_output_channel = CT_out[c, :, :, :]  # 选择第 c 个通道,并增加通道维度
        # MRI_output_channel = MRI_out[c, :, :, :].detach().cpu().numpy()  # 选择第 c 个通道,并增加通道维度
        MRI_output_channel = MRI_out[c, :, :, :]  # 选择第 c 个通道,并增加通道维度
        # 计算所有通道的平均损失
        loss_channel = loss_fun(CT_output_channel, MRI_output_channel)
        channel_losses = channel_losses + loss_channel
    mean_loss = channel_losses/ lens
    # ssim_loss = -np.log(mean_loss.detach().cpu().numpy() + BN_EPS)
    return mean_loss

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
def get_current_consistency_weight(args,cons_ramp_type, epoch):
    if cons_ramp_type == 'sig_ram':
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)
    elif cons_ramp_type == 'lin_ram':
        return args.consistency * linear_rampup(epoch, args.consistency_rampup)
    elif cons_ramp_type == 'cos_ram':
        return args.consistency * cosine_rampdown(epoch, args.consistency_rampup)

    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func1,loss_func2,loss_func3, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            CT_image, CT_seg, MRI_image, MRI_seg,label_id = batch_data["CT_image"], batch_data["CT_seg"], batch_data["MRI_image"], batch_data["MRI_seg"], batch_data["label"]
        CT_image, CT_seg, MRI_image, MRI_seg = CT_image.cuda(args.rank), CT_seg.cuda(args.rank), MRI_image.cuda(args.rank), MRI_seg.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=True):
            CT_img_F_ds, MRI_img_F_ds, CT_seg_out, MRI_seg_out = model(CT_image, MRI_image)
            consistency_weight = get_current_consistency_weight(args, 'sig_ram', epoch)
            contra_weight = get_current_consistency_weight(args, 'cos_ram', epoch)
            sup_loss, csc_loss, cac_loss = my_sup_loss(loss_func1, loss_func2, loss_func3, CT_seg_out, MRI_seg_out, CT_seg, MRI_seg, CT_img_F_ds, MRI_img_F_ds, label_id)
            loss = sup_loss + consistency_weight * csc_loss + contra_weight*cac_loss
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    args.amp = True
    epoch_dice = 0.0
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                CT_image, CT_seg, MRI_image, MRI_seg, label_id = batch_data["CT_image"], batch_data["CT_seg"], \
                batch_data["MRI_image"], batch_data["MRI_seg"], batch_data["label"]
                CT_image, CT_seg, MRI_image, MRI_seg = CT_image.cuda(args.rank), CT_seg.cuda(args.rank), MRI_image.cuda(args.rank), MRI_seg.cuda(args.rank)
            torch.cuda.empty_cache()
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    CT_img_F_ds, MRI_img_F_ds, CT_seg_out, MRI_seg_out = model(CT_image, MRI_image)
                else:
                    CT_img_F_ds, MRI_img_F_ds, CT_seg_out, MRI_seg_out = model(CT_image, MRI_image)
            # if not logits.is_cuda:
            #     target = target.cpu()
            output, target = CT_seg_out.cpu(), CT_seg.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(output)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc, not_nans = acc.cuda(args.rank), not_nans.cuda(args.rank)
            ## compute dice score
            epoch_dice = get_dice_score(CT_seg_out, CT_seg)

            if args.distributed:
                is_valid = True
                acc_list, not_nans_list = distributed_all_gather([acc, not_nans], out_numpy=True, is_valid=is_valid)
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()

    return avg_acc, epoch_dice

def get_dice_score(prev_masks, gt3D):
    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = (mask_pred > mask_threshold)
        mask_gt = (mask_gt > 0)

        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2 * volume_intersect / volume_sum

    pred_masks = (prev_masks > 0.5)
    true_masks = (gt3D > 0)
    dice_list = []
    for i in range(true_masks.shape[0]):
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
    return (sum(dice_list) / len(dice_list)).item()


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, best_dice = 0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "best_dice": best_dice, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    consistency_loss,
    contrastive_loss,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    best_dice = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        # if args.distributed:
        #     train_loader.sampler.set_epoch(epoch)
        #     torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func1=loss_func, loss_func2=consistency_loss, loss_func3=contrastive_loss, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc, epoch_dice = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_dice = np.mean(epoch_dice)
            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    "epoch_dice",
                    val_avg_acc,
                    val_avg_dice,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                    writer.add_scalar("epoch_dice", epoch_dice, epoch)
                if epoch_dice > best_dice:
                    # print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    print("new best ({:.6f} --> {:.6f}). ".format(best_dice, epoch_dice))
                    best_dice = epoch_dice
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_avg_acc, best_dice=best_dice, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_avg_acc, best_dice=best_dice, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best dice: ", best_dice)

    return val_acc_max
