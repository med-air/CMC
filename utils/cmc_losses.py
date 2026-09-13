# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# CMC custom loss functions extracted from main.py so they can be
# imported independently of the CLI entrypoint and unit-tested without
# the full training-stack dependencies (tensorboardX, monai, etc.,
# are still required because monai.losses.ContrastiveLoss is used
# inside CSC_loss).

from __future__ import annotations

import torch

# CSC_loss uses monai.losses.ContrastiveLoss — import lazily so that
# importing this module for, e.g., a CAC_loss-only test does not
# require monai.
_MONAI_CONTRASTIVE = None


def _monai_contrastive(*args, **kwargs):
    global _MONAI_CONTRASTIVE
    if _MONAI_CONTRASTIVE is None:
        from monai.losses import ContrastiveLoss  # type: ignore
        _MONAI_CONTRASTIVE = ContrastiveLoss
    return _MONAI_CONTRASTIVE(*args, **kwargs)


def CAC_loss(pred1, pred2, similarity: str = "cosine"):
    """
    Channel-wise anatomical consistency loss (CAC).

    Numerical-stability notes (Issue #10):
      * `eps` is added to every denominator to prevent zero-division
        when both predictions are empty or perfectly disjoint.
      * The fallback constant returned for the all-zero edge case is
        created with ``requires_grad=True`` so that it still
        participates in the autograd graph; a detached tensor here
        would silently sever the gradient path from ``CAC_loss`` back
        into the model.

    Args:
        pred1, pred2: tensors of identical shape (B, C, *spatial).
            ``spatial`` may be 2-D, 3-D, or N-D.
        similarity: reserved for future use; currently always 'cosine'.

    Returns:
        A scalar tensor with ``requires_grad`` propagating back into
        ``pred1`` / ``pred2`` whenever they are non-degenerate.
    """
    del similarity  # currently unused
    eps = 1e-8
    smooth = 1e-6
    # Magnitude-based predicate (rather than `==`) avoids the
    # `RuntimeError: bool value of ... is ambiguous` that arises when
    # summing floats with requires_grad=True under AMP.
    if torch.sum(pred1).abs() < eps and torch.sum(pred2).abs() < eps:
        return torch.tensor(1.0, device=pred1.device, requires_grad=True)
    dim_len = len(pred1.size())
    if dim_len == 5:
        dim = (2, 3, 4)
    elif dim_len == 4:
        dim = (2, 3)
    else:
        # Robust fall-back: sum over every non-batch, non-channel axis.
        dim = tuple(range(2, dim_len))
    intersect = torch.sum(pred1 * pred2, dim=dim)
    y_sum = torch.sum(pred1 * pred1, dim=dim)
    z_sum = torch.sum(pred2 * pred2, dim=dim)
    # Guard the denominator against degenerate (all-zero) channels.
    dice_sim = (2 * intersect + smooth) / (z_sum + y_sum + smooth + eps)
    dice_sim = dice_sim.mean()
    if torch.isnan(dice_sim):
        # IMPORTANT: keep requires_grad=True so this branch does not
        # silently sever the autograd graph (Issue #24 follow-on).
        dice_sim = torch.tensor(1.0, device=dice_sim.device, requires_grad=True)
    return dice_sim


def CSC_loss(pred1, pred2):
    """
    Channel-wise semantic consistency loss (CSC).

    Computes a per-channel MONAI ``ContrastiveLoss`` between
    ``pred1`` and ``pred2`` and averages over the batch axis.

    Numerical-stability notes (Issue #10):
      * The NaN fallback tensor is created with ``requires_grad=True``
        so the autograd graph stays connected; otherwise this branch
        would silently drop gradients to the encoder / fusion layer.
      * The accumulator is initialized as a 0-d tensor with
        ``requires_grad=True`` (not a Python ``0.0``) so that the
        first ``+= cl_value`` actually builds the graph rather than
        relying on the autograd engine to retrofit it.
    """
    eps = 1e-8
    channel_losses = torch.tensor(0.0, device=pred1.device, requires_grad=True)
    lens = pred1.shape[0]
    for c in range(lens):
        # Select the c-th slice along the batch axis. Use the bare
        # index `pred1[c]` (not `pred1[c, :, :, :]`) so this works for
        # both 4-D feature maps (B, C, H, W) and the 5-D volumetric
        # feature maps (B, C, D, H, W) produced by Semi_SM_model.
        # The original `pred1[c, :, :, :]` would raise IndexError on a
        # 5-D input — silently breaking the semi-supervised branch
        # after `start_fusion_epoch`.
        pred1_output_channel = pred1[c]  # select the c-th channel of pred1
        pred2_output_channel = pred2[c]  # select the c-th channel of pred2
        pred1_2d_flat = pred1_output_channel.reshape(
            -1, pred1_output_channel.shape[0]
        )  # resize shape
        pred2_2d_flat = pred2_output_channel.reshape(
            -1, pred2_output_channel.shape[0]
        )  # resize shape
        # compute the ContrastiveLoss from each channel
        cl_loss = _monai_contrastive(batch_size=2, temperature=0.5)
        cl_value = cl_loss(pred1_2d_flat, pred2_2d_flat)
        channel_losses = channel_losses + cl_value
    # Guard against division-by-zero if `lens` is somehow 0.
    mean_loss = channel_losses / (lens + eps)
    if torch.isnan(mean_loss):
        mean_loss = torch.tensor(1.0, device=mean_loss.device, requires_grad=True)
    return mean_loss
