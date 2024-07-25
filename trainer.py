import os
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather, get_dice_score
import gc

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
            CT_sup_loss = loss_func1(CT_seg_out.unsqueeze(1), CT_seg.unsqueeze(1))
            MRI_sup_loss = loss_func1(MRI_seg_out.unsqueeze(1), MRI_seg.unsqueeze(1))
            csc_loss = my_csc_loss(loss_func2, CT_img_F_ds[i], MRI_img_F_ds[i])
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
    BN_EPS = 1e-4  # 1e-4  #1e-5
    channel_losses = 0.0
    lens = CT_out.shape[0]
    for c in range(CT_out.shape[0]):
        CT_output_channel = CT_out[c, :, :, :]  # 选择第 c 个通道,并增加通道维度
        MRI_output_channel = MRI_out[c, :, :, :]  # 选择第 c 个通道,并增加通道维度
        # 计算所有通道的平均损失
        loss_channel = loss_fun(CT_output_channel, MRI_output_channel)
        channel_losses = channel_losses + loss_channel
    mean_loss = channel_losses/ lens
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
        if epoch < args.fusion_start_epoch:
            return args.smooth_nr 
        else:
            return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)
    elif cons_ramp_type == 'lin_ram':
        return args.consistency * linear_rampup(epoch, args.consistency_rampup)
    elif cons_ramp_type == 'cos_ram':
        return args.consistency * cosine_rampdown(epoch, args.consistency_rampup)
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242


# def train_epoch(model, fabric, loader, optimizer, scaler, epoch, loss_func1,loss_func2,loss_func3, args):
def train_epoch(model, loader, optimizer, scaler, epoch, loss_func1,loss_func2,loss_func3, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        # batch_data = fabric.to_device(batch_data)

        CT_image, CT_seg, MRI_image, MRI_seg,label_id = batch_data["CT_image"], batch_data["CT_seg"], batch_data["MRI_image"], batch_data["MRI_seg"], batch_data["label"]
        CT_image, CT_seg, MRI_image, MRI_seg = CT_image.cuda(args.rank), CT_seg.cuda(args.rank), MRI_image.cuda(args.rank), MRI_seg.cuda(args.rank)
        with autocast(enabled=True):
            CT_img_F_ds, MRI_img_F_ds, CT_seg_out, MRI_seg_out = model(CT_image, MRI_image)
            CT_img_F_ds, MRI_img_F_ds, CT_seg_out, MRI_seg_out = CT_img_F_ds.cuda(args.rank), MRI_img_F_ds.cuda(args.rank),CT_seg_out.cuda(args.rank),MRI_seg_out.cuda(args.rank)
            consistency_weight = get_current_consistency_weight(args, 'sig_ram', epoch)
            contra_weight = get_current_consistency_weight(args, 'cos_ram', epoch)
            sup_loss, csc_loss, cac_loss = my_sup_loss(loss_func1, loss_func2, loss_func3, CT_seg_out, MRI_seg_out, CT_seg, MRI_seg, CT_img_F_ds, MRI_img_F_ds, label_id)
            consistency_weight = 0
            # contra_weight = 0
            loss = sup_loss + consistency_weight * csc_loss + contra_weight*cac_loss
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # fabric.backward(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.empty_cache()

        if args.dist:
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
    gc.collect()
    torch.cuda.empty_cache()

    return run_loss.avg


def val_epoch(model, loader,save_root, epoch, dice_func, args):
    model.eval()
    args.amp = True
    epoch_dice_arr=[]

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            CT_image, CT_seg, MRI_image, MRI_seg, label_id = batch_data["CT_image"], batch_data["CT_seg"], \
            batch_data["MRI_image"], batch_data["MRI_seg"], batch_data["label"]
            CT_image, CT_seg, MRI_image, MRI_seg = CT_image.cuda(args.rank), CT_seg.cuda(args.rank), MRI_image.cuda(args.rank), MRI_seg.cuda(args.rank)
            with autocast(enabled=args.amp):
                CT_img_F_ds, MRI_img_F_ds, CT_seg_out, MRI_seg_out = model(CT_image, MRI_image)

            epoch_dice_CT = get_dice_score(CT_seg_out, CT_seg)
            epoch_dice_MRI = get_dice_score(MRI_seg_out, MRI_seg)
            epoch_dice = (epoch_dice_CT+epoch_dice_MRI)/2
            epoch_dice_arr.append(epoch_dice)
            with open(os.path.join(save_root, 'log.txt'), 'a') as f:
                print("epoch_dice:{}".format(epoch_dice), file=f)
        dice_func.reset()
    mean_dice_epoch = np.mean(epoch_dice_arr)
    with open(os.path.join(save_root, 'log.txt'), 'a') as f:
        print("mean_dice_epoch:{}".format(mean_dice_epoch), file=f)
    return mean_dice_epoch
def save_checkpoint(model, epoch, args, optimizer, scheduler,filename="model.pt"):
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "epoch": epoch
    }
    checkpoint_filename = os.path.join(args.logdir, filename)
    torch.save(checkpoint, checkpoint_filename)
    print("Saving checkpoint", checkpoint_filename)
def run_training(
    model,
    # fabric,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    consistency_loss,
    contrastive_loss,
    dice_metric,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    best_dice = 0.0
    save_root = os.path.join(args.logdir,args.backbone)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for epoch in range(start_epoch, args.max_epochs):
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func1=loss_func, loss_func2=consistency_loss, loss_func3=contrastive_loss, args=args
        )
        print(
            "Final training  {}/{}".format(epoch, args.max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        with open(os.path.join(save_root, 'log.txt'), 'a') as f:
            print("Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),file=f)
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.dist:
                torch.distributed.barrier()
            epoch_time = time.time()
            mean_dice_val= val_epoch(
                model,
                val_loader,
                save_root,
                epoch=epoch,
                dice_func=dice_metric,
                args=args
            )
            if writer is not None:
                writer.add_scalar("mean_dice_val", mean_dice_val, epoch)
            if epoch == args.max_epochs:
                save_checkpoint(model, epoch, args, optimizer, scheduler, filename="final_model.pt")
            if mean_dice_val > best_dice:
                print("Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                      "mean_dice_val",
                      mean_dice_val,
                      "time {:.2f}s".format(time.time() - epoch_time))
                with open(os.path.join(save_root, 'log.txt'), 'a') as f:
                    print("Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                          "mean_dice_val",
                          mean_dice_val,
                          "time {:.2f}s".format(time.time() - epoch_time), file=f)
                    print("#################### new best ({:.6f} --> {:.6f}). ".format(best_dice, mean_dice_val),
                          file=f)
                best_dice = mean_dice_val
                save_checkpoint(model, epoch, args, optimizer, scheduler, filename="best_model.pt")
                with open(os.path.join(save_root, 'log.txt'), 'a') as f:
                    print("save the best model!!!!, best_dice", best_dice, file=f)

        if scheduler is not None:
            scheduler.step()
    print("Training Finished !, Best dice: ", best_dice)

