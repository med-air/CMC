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
from utils.utils import dice, resample_3d, ORGAN_NAME

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

def train_epoch(model, unlabeled_model, ct_loader, mri_loader, ct_unlabeled_loader, mri_unlabeled_loader, optimizer, scaler, epoch, loss_func, CSC_loss_func, CAC_loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    CSC_loss = 0.0
    CAC_loss = 0.0
    save_log_dir = args.logdir
    for idx, (batch_ct, batch_mri, batch_ct_unlabeled, batch_mri_unlabeled) in enumerate(zip(ct_loader, mri_loader,ct_unlabeled_loader, mri_unlabeled_loader)):
        if isinstance(batch_ct, list) and isinstance(batch_ct, list) and isinstance(batch_ct_unlabeled, list) and isinstance(batch_mri_unlabeled, list):
            ct_data, ct_target = batch_ct
            mri_data, mri_target = batch_mri
            ct_unlabeled_data, mri_unlabeled_data = batch_ct_unlabeled, batch_mri_unlabeled
        else:
            ct_data, ct_target = batch_ct["img_CT"], batch_ct["mask_CT"]
            mri_data, mri_target = batch_mri["img_MRI"], batch_mri["mask_MRI"]
            ct_unlabeled_data, mri_unlabeled_data = batch_ct_unlabeled["img_CT"], batch_mri_unlabeled["img_MRI"]
        ct_data, ct_target, mri_data, mri_target = ct_data.cuda(args.rank), ct_target.cuda(args.rank), mri_data.cuda(args.rank), mri_target.cuda(args.rank)
        ct_unlabeled_data, mri_unlabeled_data = ct_unlabeled_data.cuda(args.rank), mri_unlabeled_data.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            _, _, ct_logits, mri_logits = model(ct_data, mri_data)
            ct_loss = loss_func(ct_logits, ct_target)
            mri_loss = loss_func(mri_logits, mri_target)
            sup_loss = (ct_loss + mri_loss)/2
            loss = sup_loss
        if epoch >= args.start_fusion_epoch:
            ### start semi-supervised on unlabeled data
            model.cuda()
            model.eval()
            with autocast(enabled=args.amp):
                with torch.no_grad():
                    ct_img_F_ds, mri_img_F_ds, ct_unlabeled_output, mri_unlabeled_output = model(ct_unlabeled_data, mri_unlabeled_data)
                ### compute CSC loss
                CSC_loss = CSC_loss_func(ct_img_F_ds, mri_img_F_ds)
                CAC_loss = CAC_loss_func(ct_unlabeled_output, mri_unlabeled_output)
                ### compute CSC loss
                consistency_weight_csc = sigmoid_rampup(epoch, args.max_epochs)
                ### compute CAC loss
                consistency_weight_cac = cosine_rampdown(epoch, args.max_epochs)
                loss = sup_loss + consistency_weight_csc * CSC_loss + consistency_weight_cac * CAC_loss

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < ct_loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(ct_loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "CSC_loss: {:.4f}".format(CSC_loss),
                "CAC_loss: {:.4f}".format(CAC_loss),
                "time {:.2f}s".format(time.time() - start_time),
            )
            with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(ct_loader)),
                    "loss: {:.4f}".format(run_loss.avg),
                    "CSC_loss: {:.4f}".format(CSC_loss),
                    "CAC_loss: {:.4f}".format(CAC_loss),
                    "time {:.2f}s".format(time.time() - start_time),file=f
                )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, ct_loader, mri_loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    all_avg_dice =[]
    start_time = time.time()
    model_inferer =None
    save_log_dir = args.logdir
    nun_class = args.out_channels
    with torch.no_grad():
        for idx, (batch_ct, batch_mri) in enumerate(zip(ct_loader,mri_loader)):
            if isinstance(batch_ct, list) and isinstance(batch_mri, list):
                ct_data, ct_target = batch_ct
                mri_data, mri_target = batch_mri
            else:
                ct_data, ct_target = batch_ct["img_CT"], batch_ct["mask_CT"]
                mri_data, mri_target = batch_mri["img_MRI"], batch_mri["mask_MRI"]
            ct_data, ct_target, mri_data, mri_target = ct_data.cuda(args.rank), ct_target.cuda(
                args.rank), mri_data.cuda(args.rank), mri_target.cuda(args.rank)
            _, _, h, w, d = ct_target.shape
            target_shape = (h, w, d)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    _, _, ct_logits, mri_logits = model_inferer(ct_data, mri_data)
                else:
                    _, _, ct_logits, mri_logits= model(ct_data, mri_data)
            if not ct_logits.is_cuda:
                ct_target, mri_target = ct_target.cpu(), mri_target.cpu()
            val_outputs = torch.softmax(ct_logits, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels = ct_target.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)
            # img_name = batch_ct["img_CT_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            organ_dice = []
            for i in range(1, nun_class):
                organ_name = ORGAN_NAME[i-1]
                if organ_name == 'Spleen':
                    spleen_dice = dice(val_outputs == i, val_labels == i)
                    organ_dice.append(spleen_dice)
                    print("spleen dice:",spleen_dice)
                elif organ_name == 'Right Kidney':
                    R_kidney_dice = dice(val_outputs == i, val_labels == i)
                    organ_dice.append(R_kidney_dice)
                    print("Right Kidney dice:",R_kidney_dice)
                elif organ_name == 'Left Kidney':
                    L_kidney_dice = dice(val_outputs == i, val_labels == i)
                    organ_dice.append(L_kidney_dice)
                    print("Left Kidney dice:",L_kidney_dice)
                elif organ_name == 'Liver':
                    Liver_dice = dice(val_outputs == i, val_labels == i)
                    organ_dice.append(Liver_dice)
                    print("Liver dice:",Liver_dice)
                elif i>8:
                    break
            avg_dice = np.mean(organ_dice)
            print("avg_dice:{}".format(avg_dice))
            all_avg_dice.append(avg_dice)
            if args.rank == 0:
                avg_acc = avg_dice
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(ct_loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
                with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                    print(
                        "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(ct_loader)),
                        "acc",
                        avg_acc,
                        "time {:.2f}s".format(time.time() - start_time),file=f
                    )
            start_time = time.time()
    return np.mean(all_avg_dice)

def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    unlabeled_model,
    ct_train_loader,
    ct_val_loader,
    mri_train_loader,
    mri_val_loader,
    ct_unlabeled_loader,
    mri_unlabeled_loader,
    optimizer,
    loss_func,
    CSC_loss,
    CAC_loss,
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
    save_log_dir = args.logdir
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            ct_train_loader.sampler.set_epoch(epoch)
            mri_train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, unlabeled_model, ct_train_loader,mri_train_loader, ct_unlabeled_loader, mri_unlabeled_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, CSC_loss_func=CSC_loss, CAC_loss_func=CAC_loss, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                print(
                    "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                    "loss: {:.4f}".format(train_loss),
                    "time {:.2f}s".format(time.time() - epoch_time),file=f
                )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0 or (epoch + 1)== args.max_epochs:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                ct_val_loader,
                mri_val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                    print(
                        "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                        "acc",
                        val_avg_acc,
                        "time {:.2f}s".format(time.time() - epoch_time),file=f
                    )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                        print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc),file=f)
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                        print("Copying to model.pt new best model!!!!",file=f)
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))
        if scheduler is not None:
            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
        print("Training Finished !, Best Accuracy: ", val_acc_max,file=f)

    return val_acc_max
