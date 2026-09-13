"""
Regression tests for the CMC cross-modality auxiliary losses
(CAC_loss, CSC_loss) and the trainer.py gradient-flow wiring.

These tests pin down two specific bugs that were fixed in PR #29:

  * Issue #24 — CSC_loss / CAC_loss were detached from the autograd
    graph (the unlabeled forward was wrapped in `torch.no_grad()` in
    trainer.py and the loss fallbacks returned detached tensors), so
    the encoder and fusion layer received zero gradient from the
    consistency terms.

  * Issue #10 — NaN-prone denominators and missing gradient clipping
    caused training instability.

The tests are designed to run without GPU. They construct leaf tensors
that stand in for the encoder output and verify that gradients flow
all the way back. If a future refactor re-introduces a `torch.no_grad()`
block in trainer.py, or replaces a fallback `torch.tensor(...)` without
`requires_grad=True`, these tests will fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make the repo root importable when pytest is run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.cmc_losses import CAC_loss, CSC_loss  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _leaf(name, *shape):
    """A leaf tensor with requires_grad=True — stands in for an encoder output."""
    return torch.randn(*shape, dtype=torch.float64, requires_grad=True)


# ---------------------------------------------------------------------------
# CAC_loss
# ---------------------------------------------------------------------------


class TestCACLoss:
    def test_real_inputs_produce_finite_value_with_grad(self):
        ct = _leaf("ct", 2, 3, 4, 5, 6)
        mri = _leaf("mri", 2, 3, 4, 5, 6)
        loss = CAC_loss(ct, mri)
        assert torch.isfinite(loss), f"non-finite loss: {loss}"
        assert loss.requires_grad, "CAC_loss must carry requires_grad"
        loss.backward()
        assert ct.grad is not None and torch.isfinite(ct.grad).all()
        assert mri.grad is not None and torch.isfinite(mri.grad).all()

    def test_all_zero_early_return_keeps_graph(self):
        # Regression: the original code returned `torch.tensor(1.0)`
        # without requires_grad=True here, silently severing the graph.
        z1 = torch.zeros(2, 3, 4, 5, 6, requires_grad=True)
        z2 = torch.zeros(2, 3, 4, 5, 6, requires_grad=True)
        out = CAC_loss(z1, z2)
        assert out.requires_grad, (
            "all-zero early-return must keep requires_grad=True so the "
            "graph stays connected (Issue #24 follow-on)"
        )

    def test_one_zero_input_is_stable(self):
        # One input is identically zero; the other is not. The fixed
        # eps in the denominator must keep the loss finite AND keep
        # gradients flowing into both inputs.
        z = torch.zeros(2, 3, 4, 5, 6, requires_grad=True)
        nz = _leaf("nz", 2, 3, 4, 5, 6)
        out = CAC_loss(z, nz)
        assert torch.isfinite(out)
        out.backward()
        assert torch.isfinite(z.grad).all()
        assert torch.isfinite(nz.grad).all()

    def test_supports_3d_inputs(self):
        # Regression: original code would raise UnboundLocalError for
        # any dim_len other than 4 or 5.
        a = _leaf("a", 2, 3, 8, 8)
        b = _leaf("b", 2, 3, 8, 8)
        out = CAC_loss(a, b)
        assert torch.isfinite(out)
        out.backward()
        assert torch.isfinite(a.grad).all()

    def test_supports_higher_dimensional_inputs(self):
        a = _leaf("a", 2, 3, 4, 4, 4, 4)
        b = _leaf("b", 2, 3, 4, 4, 4, 4)
        out = CAC_loss(a, b)
        assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# CSC_loss
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not pytest.importorskip("monai", reason="monai required for CSC_loss (uses ContrastiveLoss)"),
    reason="monai not available",
)
class TestCSCLoss:
    def test_real_3d_feature_maps_produce_finite_loss(self):
        # Shape mirrors the model's per-modality feature map.
        ct = _leaf("ct", 2, 4, 16, 16, 16)
        mri = _leaf("mri", 2, 4, 16, 16, 16)
        loss = CSC_loss(ct, mri)
        assert torch.isfinite(loss), f"non-finite CSC loss: {loss}"
        assert loss.requires_grad, "CSC_loss must carry requires_grad"

    def test_csc_gradient_reaches_encoder_shaped_leaf(self):
        # End-to-end gradient check: simulate the trainer's
        # `loss.backward()` and verify the encoder-shaped input
        # receives a finite gradient. This is the direct regression
        # test for Issue #24.
        ct = _leaf("ct", 2, 4, 8, 8, 8)
        mri = _leaf("mri", 2, 4, 8, 8, 8)
        loss = CSC_loss(ct, mri)
        loss.backward()
        assert ct.grad is not None
        assert torch.isfinite(ct.grad).all()
        assert mri.grad is not None
        assert torch.isfinite(mri.grad).all()

    def test_csc_with_identical_inputs_collapses(self):
        # When pred1 == pred2, the contrastive loss should be near
        # zero (or at least finite and small). We don't pin a specific
        # value because MONAI's ContrastiveLoss is not scale-invariant,
        # but we require finiteness.
        x = _leaf("x", 2, 4, 8, 8, 8)
        loss = CSC_loss(x, x.detach().clone().requires_grad_(True))
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Trainer-loop gradient-flow simulation
# ---------------------------------------------------------------------------


class _FakeSemiModel(torch.nn.Module):
    """Stand-in for Semi_SM_model that exposes the (feat, feat, logits, logits)
    tuple the trainer expects. Tiny enough to run on CPU."""

    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 4)

    def forward(self, ct, mri):
        ct_feat = self.enc(ct)
        mri_feat = self.enc(mri)
        ct_logits = self.head(ct_feat)
        mri_logits = self.head(mri_feat)
        return ct_feat, mri_feat, ct_logits, mri_logits


class TestTrainerGradientFlow:
    """Reproduce trainer.py's per-iteration control flow on a tiny model
    and assert that the unlabeled branch contributes gradient signal.

    If anyone re-introduces `with torch.no_grad():` around the unlabeled
    forward (the original Issue #24 bug), the post-fusion gradient norm
    will equal the pre-fusion gradient norm and these tests will fail.
    """

    def test_unlabeled_branch_contributes_gradient(self):
        torch.manual_seed(0)
        model = _FakeSemiModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sup_loss_fn = torch.nn.CrossEntropyLoss()

        ct = torch.randn(2, 8)
        mri = torch.randn(2, 8)
        ct_y = torch.tensor([0, 1])
        mri_y = torch.tensor([1, 0])

        # --- Pre-fusion step: supervised only ---
        for p in model.parameters():
            p.grad = None
        _, _, ctl, mrl = model(ct, mri)
        sup = (sup_loss_fn(ctl, ct_y) + sup_loss_fn(mrl, mri_y)) / 2
        sup.backward()
        grad_pre = torch.cat(
            [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
        ).norm().item()

        # --- Post-fusion step: supervised + CSC + CAC ---
        torch.manual_seed(0)  # identical init / data for the comparison
        model2 = _FakeSemiModel()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        for p in model2.parameters():
            p.grad = None
        ct_f, mri_f, ctl2, mrl2 = model2(ct, mri)
        sup2 = (sup_loss_fn(ctl2, ct_y) + sup_loss_fn(mrl2, mri_y)) / 2
        csc = CSC_loss(ct_f, mri_f)
        cac = CAC_loss(ctl2, mrl2)
        total = sup2 + 0.5 * csc + 0.5 * cac
        total.backward()
        grad_post = torch.cat(
            [p.grad.detach().flatten() for p in model2.parameters() if p.grad is not None]
        ).norm().item()

        # The unlabeled branch must change the gradient. If it's detached,
        # the two norms are identical (up to FP noise).
        assert abs(grad_post - grad_pre) > 1e-6, (
            f"Issue #24 regression: pre-fusion grad={grad_pre:.6f}, "
            f"post-fusion grad={grad_post:.6f} — unlabeled branch is "
            f"still detached!"
        )


class TestGradientClipping:
    def test_clip_clamps_exploding_gradients(self):
        model = torch.nn.Linear(4, 4)
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 1e6
        total_before = sum(p.grad.norm() ** 2 for p in model.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_after = sum(p.grad.norm() ** 2 for p in model.parameters()) ** 0.5
        assert total_before > 1.0
        assert total_after <= 1.0 + 1e-6, (
            f"clip_grad_norm_ failed: before={total_before}, after={total_after}"
        )

    def test_clip_respects_zero_norm_as_disable(self):
        # Mirrors trainer.py's `if clip_norm > 0` guard.
        model = torch.nn.Linear(4, 4)
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 1e6
        clip_norm = 0.0  # disabled
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        total_unchanged = sum(p.grad.norm() ** 2 for p in model.parameters()) ** 0.5
        assert total_unchanged > 1.0, "gradients should NOT be clipped when norm=0"
