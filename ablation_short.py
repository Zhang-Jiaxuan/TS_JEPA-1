# ablation_short.py
# Place beside koopman_short.py at the TS_JEPA repository root.
#
# Variants:
#   A_matched: same two-branch architecture and initial spectral parameters
#              as B, but both blocks use the unrestricted shared radius
#              interval [0.65, 0.999].
#   Single:    one encoder branch + unconstrained MLP latent predictor;
#              no Koopman blocks or spectral-radius constraints.
#
# Data preprocessing, windows, and downstream residual linear head follow
# the previous koopman_short.py experimental protocol.

import argparse
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import koopman_short as base
from src.models.encoder import Encoder


class MatchedInitModel(base.KoopmanTSJEPA):
    def __init__(self, dim=128, slow_dim=32, fast_dim=32, local_window=3):
        # Creates precisely the same shared encoder and two projections
        # as B. Then replaces ONLY the two radius parameterizations.
        super().__init__(
            variant="B",
            dim=dim,
            slow_dim=slow_dim,
            fast_dim=fast_dim,
            local_window=local_window,
        )
        self.slow_k = base.SpectralBlock(
            slow_dim,
            low=0.65,
            high=0.999,
            initial_radius=0.995,
            initial_angle=0.08,
        )
        self.fast_k = base.SpectralBlock(
            fast_dim,
            low=0.65,
            high=0.999,
            initial_radius=0.82,
            initial_angle=0.25,
        )
        # Inherited states() treats all non-C variants as the same
        # two-branch model. Inherited pretrain_loss() also works.
        self.variant = "A_matched"

    def features(self, history):
        slow, fast, _ = self.states(history, base.CONTEXT)
        return torch.cat(
            (self.slow_k(slow), self.fast_k(fast)), dim=-1
        )


class SingleBranchModel(nn.Module):
    def __init__(self, dim=128, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(
            num_patches=base.PRETRAIN_PATCHES,
            dim_in=base.PATCH,
            kernel_size=3,
            embed_dim=dim,
            embed_bias=True,
            nhead=2,
            num_layers=1,
            jepa=True,
        )
        self.projection = nn.Linear(dim, latent_dim)

        # An unconstrained nonlinear predictive head: no eigenvalue,
        # Koopman operator, or slow/fast spectral assignment.
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def state(self, patches, length):
        if not 1 <= length <= patches.size(1):
            raise ValueError("Invalid observed prefix length.")
        tokens = self.encoder(patches[:, :length, :])
        return self.projection(tokens.mean(dim=1))

    def rollout(self, z, steps):
        for _ in range(steps):
            z = self.predictor(z)
        return z

    def features(self, history):
        z = self.state(history, base.CONTEXT)
        return self.predictor(z)


def build_model(args):
    if args.variant == "A_matched":
        return MatchedInitModel(
            dim=args.dim,
            slow_dim=args.slow_dim,
            fast_dim=args.fast_dim,
            local_window=args.local_window,
        )
    return SingleBranchModel(
        dim=args.dim,
        latent_dim=args.slow_dim + args.fast_dim,
    )


def load_initial_encoder(model, checkpoint_path):
    if not checkpoint_path:
        return

    saved = torch.load(checkpoint_path, map_location="cpu")
    weights = saved["encoder"] if "encoder" in saved else saved

    # The number of patches may differ from the old checkpoint.
    # The new 13-position sinusoidal encoding has already been created.
    weights = {
        key: value for key, value in weights.items()
        if key != "pos_embed"
    }
    incompatible = model.encoder.load_state_dict(
        weights, strict=False
    )
    if (
        set(incompatible.missing_keys) != {"pos_embed"}
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            f"Incompatible encoder checkpoint: {incompatible}"
        )


def single_pretrain_loss(model, teacher, patches, context_len):
    state = model.state(patches, context_len)

    # Each teacher target includes only its own observed prefix.
    with torch.no_grad():
        targets = [
            teacher.state(patches, context_len + k)
            for k in range(1, base.MAX_STEP + 1)
        ]

    latent_loss = torch.stack([
        F.smooth_l1_loss(
            model.rollout(state, k),
            target,
        )
        for k, target in enumerate(targets, start=1)
    ]).mean()

    # Match the anti-collapse coefficient of the two-branch script.
    loss = latent_loss + 0.05 * base.variance_floor(state)
    return loss


def compute_loss(model, teacher, patches, context_len, variant):
    if variant == "A_matched":
        # Exactly the existing B objective and coefficients.
        loss, _ = base.pretrain_loss(
            model, teacher, patches, context_len
        )
        return loss
    return single_pretrain_loss(
        model, teacher, patches, context_len
    )


def pretrain(args, device):
    train, val, _ = base.make_loaders(
        args.csv,
        base.PRETRAIN_PATCHES,
        args.batch_size,
    )

    model = build_model(args).to(device)
    load_initial_encoder(model, args.init_encoder)

    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,
    )
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for patches in train:
            patches = patches.to(device)
            context_len = random.randint(4, base.CONTEXT)

            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(
                model, teacher, patches,
                context_len, args.variant
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

            with torch.no_grad():
                for online_p, teacher_p in zip(
                    model.parameters(),
                    teacher.parameters(),
                ):
                    teacher_p.lerp_(
                        online_p, 1.0 - args.ema
                    )

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for patches in val:
                patches = patches.to(device)
                loss = compute_loss(
                    model, teacher, patches,
                    base.CONTEXT, args.variant
                )
                val_losses.append(loss.item())

        val_score = float(np.mean(val_losses))

        if args.variant == "A_matched":
            spectrum = (
                f"rho_s={model.slow_k.radii().mean().item():.4f} "
                f"rho_f={model.fast_k.radii().mean().item():.4f}"
            )
        else:
            spectrum = "rho_s=N/A rho_f=N/A"

        print(
            f"variant={args.variant} epoch={epoch:03d} "
            f"train={np.mean(train_losses):.6f} "
            f"val={val_score:.6f} {spectrum}"
        )

        if val_score < best_val:
            best_val = val_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "variant": args.variant,
                    "dim": args.dim,
                    "slow_dim": args.slow_dim,
                    "fast_dim": args.fast_dim,
                    "local_window": args.local_window,
                    "pretrain_epoch": epoch,
                    "val_pretrain_loss": best_val,
                },
                args.ckpt,
            )


def evaluate_downstream(args, device):
    train, val, test = base.make_loaders(
        args.csv,
        base.CONTEXT + 1,
        args.batch_size,
    )

    saved = torch.load(
        args.ckpt, map_location="cpu"
    )
    if saved["variant"] != args.variant:
        raise ValueError(
            "Checkpoint variant and --variant do not match."
        )

    # Instantiate with the saved dimensions, not potentially different
    # command-line defaults.
    saved_args = argparse.Namespace(
        variant=saved["variant"],
        dim=saved["dim"],
        slow_dim=saved["slow_dim"],
        fast_dim=saved["fast_dim"],
        local_window=saved["local_window"],
    )
    model = build_model(saved_args).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Both new variants provide 64-dimensional features by default,
    # matching B's slow_dim + fast_dim.
    feature_dim = saved["slow_dim"] + saved["fast_dim"]
    head = nn.Linear(
        feature_dim, base.PATCH
    ).to(device)

    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.head_lr,
    )

    def run_test(loader):
        head.eval()
        squared = 0.0
        absolute = 0.0
        count = 0

        with torch.no_grad():
            for patches in loader:
                patches = patches.to(device)
                history = patches[:, :base.CONTEXT]
                target = patches[:, base.CONTEXT]
                features = model.features(history)
                prediction = (
                    history[:, -1, -1:]
                    + head(features)
                )

                error = prediction - target
                squared += error.square().sum().item()
                absolute += error.abs().sum().item()
                count += target.numel()

        return squared / count, absolute / count

    best_val = float("inf")
    best_head = None
    best_epoch = None

    for epoch in range(1, args.head_epochs + 1):
        head.train()

        for patches in train:
            patches = patches.to(device)
            history = patches[:, :base.CONTEXT]
            target = patches[:, base.CONTEXT]

            with torch.no_grad():
                features = model.features(history)

            prediction = (
                history[:, -1, -1:]
                + head(features)
            )
            loss = F.mse_loss(prediction, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_mse, val_mae = run_test(val)

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_head = copy.deepcopy(
                head.state_dict()
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"variant={args.variant} "
                f"head_epoch={epoch:03d} "
                f"val_MSE={val_mse:.6f} "
                f"val_MAE={val_mae:.6f}"
            )

    head.load_state_dict(best_head)
    test_mse, test_mae = run_test(test)

    print(
        f"{args.variant} | "
        f"best_head_epoch={best_epoch} | "
        f"best_val_MSE={best_val:.6f} | "
        f"test_MSE={test_mse:.6f} | "
        f"test_MAE={test_mae:.6f} | "
        f"test_windows={len(test.dataset)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["pretrain", "eval"],
        required=True,
    )
    parser.add_argument(
        "--variant",
        choices=["A_matched", "Single"],
        required=True,
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--init_encoder", default="")

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slow_dim", type=int, default=32)
    parser.add_argument("--fast_dim", type=int, default=32)
    parser.add_argument("--local_window", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--head_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--ema", type=float, default=0.998)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base.seed_all(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if args.mode == "pretrain":
        pretrain(args, device)
    else:
        evaluate_downstream(args, device)


if __name__ == "__main__":
    main()