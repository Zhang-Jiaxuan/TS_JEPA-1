# koopman_short.py
# 放在 https://github.com/Sennadir/TS_JEPA 仓库根目录运行。
#
# A_prime: 双分支 + Koopman 块，但两个块具有相同的谱范围和初始化；
# B:       双分支 + 人为指定的慢/快谱分离；
# C:       B + 快分支的局部观测注入。
#
# 注意：这是三者之间的受控实验脚本，不是原仓库 A 的严格复现。

import argparse
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from src.models.encoder import Encoder


PATCH = 32
CONTEXT = 10
MAX_STEP = 3
PRETRAIN_PATCHES = CONTEXT + MAX_STEP


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PatchWindows(Dataset):
    """时间窗口；窗口不会跨越 train/val/test 的分界。"""

    def __init__(self, values, patch_count, stride=PATCH):
        needed = patch_count * PATCH
        if len(values) < needed:
            raise ValueError(
                f"该数据段只有 {len(values)} 个点，至少需要 {needed} 个。"
            )
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.patch_count = patch_count
        self.starts = range(0, len(values) - needed + 1, stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        end = start + self.patch_count * PATCH
        return self.values[start:end].reshape(self.patch_count, PATCH)


def make_loaders(csv_path, patch_count, batch_size):
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.sort_values("date")
    if "OT" not in df.columns:
        raise ValueError("本脚本使用原仓库的单变量 OT 列。")

    x = df["OT"].to_numpy(dtype=np.float32)

    n_val = int(0.05 * len(x))
    n_test = int(0.30 * len(x))
    n_train = len(x) - n_val - n_test

    train_raw = x[:n_train]
    mean = float(train_raw.mean())
    std = max(float(train_raw.std()), 1e-6)

    # 三个数据段统一使用训练段统计量，避免测试段统计量泄漏。
    train = (train_raw - mean) / std
    val = (x[n_train:n_train + n_val] - mean) / std
    test = (x[n_train + n_val:] - mean) / std

    def build(values, shuffle):
        return DataLoader(
            PatchWindows(values, patch_count),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
        )

    return build(train, True), build(val, False), build(test, False)


class SpectralBlock(nn.Module):
    """
    独立的二维旋转-缩放块。
    每个二维块的特征值模长等于对应的 r；不需要运行时求特征值。
    """

    def __init__(self, dim, low, high, initial_radius, initial_angle):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("slow_dim 和 fast_dim 必须是偶数。")
        if not low < initial_radius < high:
            raise ValueError("初始半径必须严格位于给定谱区间内。")

        self.low = low
        self.high = high

        p = (initial_radius - low) / (high - low)
        initial_logit = np.log(p / (1.0 - p))

        self.radius_logit = nn.Parameter(
            torch.full((dim // 2,), float(initial_logit))
        )
        self.angle = nn.Parameter(
            torch.full((dim // 2,), float(initial_angle))
        )

    def radii(self):
        return self.low + (
            self.high - self.low
        ) * self.radius_logit.sigmoid()

    def forward(self, z):
        pairs = z.reshape(*z.shape[:-1], -1, 2)
        a = pairs[..., 0]
        b = pairs[..., 1]

        r = self.radii()
        c = self.angle.cos()
        s = self.angle.sin()

        out = torch.stack(
            (r * (c * a - s * b), r * (s * a + c * b)),
            dim=-1,
        )
        return out.flatten(-2)

    def rollout(self, z, steps):
        for _ in range(steps):
            z = self(z)
        return z


class KoopmanTSJEPA(nn.Module):
    def __init__(
        self,
        variant="A_prime",
        dim=128,
        slow_dim=32,
        fast_dim=32,
        local_window=3,
    ):
        super().__init__()

        if variant not in ("A_prime", "B", "C"):
            raise ValueError("variant 必须是 A_prime、B 或 C。")
        if not 1 <= local_window <= CONTEXT:
            raise ValueError("local_window 必须在 1 到 10 之间。")

        self.variant = variant
        self.local_window = local_window

        # 复用原仓库的 tokenizer + Transformer Encoder。
        self.encoder = Encoder(
            num_patches=PRETRAIN_PATCHES,
            dim_in=PATCH,
            kernel_size=3,
            embed_dim=dim,
            embed_bias=True,
            nhead=2,
            num_layers=1,
            jepa=True,
        )

        self.slow_proj = nn.Linear(dim, slow_dim)
        self.fast_proj = nn.Linear(dim, fast_dim)

        if variant == "A_prime":
            # 不人为规定哪一支长记忆、哪一支短记忆。
            # 两个块的谱范围和初始化完全相同。
            self.slow_k = SpectralBlock(
                slow_dim,
                low=0.65,
                high=0.999,
                initial_radius=0.95,
                initial_angle=0.10,
            )
            self.fast_k = SpectralBlock(
                fast_dim,
                low=0.65,
                high=0.999,
                initial_radius=0.95,
                initial_angle=0.10,
            )
        else:
            # B/C：明确指定慢、快两个谱区间。
            self.slow_k = SpectralBlock(
                slow_dim,
                low=0.98,
                high=0.999,
                initial_radius=0.995,
                initial_angle=0.08,
            )
            self.fast_k = SpectralBlock(
                fast_dim,
                low=0.65,
                high=0.90,
                initial_radius=0.82,
                initial_angle=0.25,
            )

        if variant == "C":
            self.inject = nn.Linear(
                fast_dim, fast_dim, bias=False
            )
            nn.init.eye_(self.inject.weight)
            with torch.no_grad():
                self.inject.weight.mul_(0.2)

    def local_observation(self, patches, length):
        left = max(0, length - self.local_window)

        # 快分支最多看到最近 local_window 个已观测 patch。
        local_tokens = self.encoder(patches[:, left:length, :])
        return self.fast_proj(local_tokens[:, -1, :])

    def states(self, patches, length):
        if length < 1 or length > patches.size(1):
            raise ValueError("非法的已观测前缀长度。")

        # 慢分支看到完整的已观测前缀。
        global_tokens = self.encoder(patches[:, :length, :])
        slow = self.slow_proj(global_tokens.mean(dim=1))

        if self.variant != "C":
            fast = self.local_observation(patches, length)
            return slow, fast, None

        # C：每次吸收一个新的局部观测。
        # 计算 u_j 时，只使用截至 j 的已观测 patch。
        fast = torch.zeros(
            patches.size(0),
            self.fast_proj.out_features,
            device=patches.device,
            dtype=patches.dtype,
        )
        innovations = []

        for j in range(1, length + 1):
            observation = self.local_observation(patches, j)
            u = self.inject(observation)
            fast = self.fast_k(fast) + u
            innovations.append(u)

        return slow, fast, torch.stack(innovations, dim=1)


def variance_floor(x, target_std=0.5):
    if x.size(0) < 2:
        return x.new_zeros(())

    std = torch.sqrt(
        x.var(dim=0, unbiased=False) + 1e-4
    )
    return F.relu(target_std - std).mean()


def cross_correlation_penalty(slow, fast):
    if slow.size(0) < 2:
        return slow.new_zeros(())

    a = slow - slow.mean(dim=0, keepdim=True)
    b = fast - fast.mean(dim=0, keepdim=True)

    a = a / torch.sqrt(
        a.var(dim=0, unbiased=False, keepdim=True) + 1e-4
    )
    b = b / torch.sqrt(
        b.var(dim=0, unbiased=False, keepdim=True) + 1e-4
    )

    cross = a.T @ b / a.size(0)
    return cross.square().mean()


def pretrain_loss(model, teacher, patches, context_len):
    slow, fast, innovations = model.states(
        patches, context_len
    )

    # EMA teacher 的目标分别来自对应长度的前缀。
    with torch.no_grad():
        targets = [
            teacher.states(patches, context_len + k)
            for k in range(1, MAX_STEP + 1)
        ]

    slow_loss = torch.stack(
        [
            F.smooth_l1_loss(
                model.slow_k.rollout(slow, k),
                target[0],
            )
            for k, target in enumerate(targets, start=1)
        ]
    ).mean()

    # A_prime/B/C 均沿用上一版定义。
    # 注意：C 的下一时刻 target fast 已包含 u_{t+1}，
    # 而预测项不能观察 u_{t+1}。因此 C 的该项仍是探索性目标。
    fast_loss = F.smooth_l1_loss(
        model.fast_k(fast),
        targets[0][1],
    )

    anti_collapse = (
        variance_floor(slow) + variance_floor(fast)
    )
    separation = cross_correlation_penalty(
        slow, fast
    )

    loss = (
        slow_loss
        + fast_loss
        + 0.05 * anti_collapse
        + 0.01 * separation
    )

    if model.variant == "C":
        loss = loss + 1e-3 * innovations.square().mean()

    return loss, (slow_loss.item(), fast_loss.item())


def train_pretrain(args, device):
    train, val, _ = make_loaders(
        args.csv,
        PRETRAIN_PATCHES,
        args.batch_size,
    )

    model = KoopmanTSJEPA(
        variant=args.variant,
        dim=args.dim,
        slow_dim=args.slow_dim,
        fast_dim=args.fast_dim,
        local_window=args.local_window,
    ).to(device)

    if args.init_encoder:
        saved = torch.load(
            args.init_encoder, map_location="cpu"
        )
        weights = (
            saved["encoder"]
            if "encoder" in saved
            else saved
        )

        # 旧 checkpoint 可能使用不同的 num_patches。
        # pos_embed 为固定正弦编码，因此保留本模型新建的版本。
        weights = {
            k: v for k, v in weights.items()
            if k != "pos_embed"
        }
        incompatible = model.encoder.load_state_dict(
            weights, strict=False
        )
        if (
            set(incompatible.missing_keys) != {"pos_embed"}
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                f"原编码器 checkpoint 不兼容：{incompatible}"
            )

    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

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

            # context_len + MAX_STEP <= PRETRAIN_PATCHES。
            context_len = random.randint(4, CONTEXT)

            optimizer.zero_grad(set_to_none=True)
            loss, _ = pretrain_loss(
                model, teacher, patches, context_len
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

            # 更新 EMA teacher，包括谱块参数。
            with torch.no_grad():
                for online_p, ema_p in zip(
                    model.parameters(),
                    teacher.parameters(),
                ):
                    ema_p.lerp_(
                        online_p, 1.0 - args.ema
                    )

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for patches in val:
                patches = patches.to(device)
                loss, _ = pretrain_loss(
                    model, teacher, patches, CONTEXT
                )
                val_losses.append(loss.item())

        val_score = float(np.mean(val_losses))

        print(
            f"variant={args.variant} "
            f"epoch={epoch:03d} "
            f"train={np.mean(train_losses):.6f} "
            f"val={val_score:.6f} "
            f"rho_s={model.slow_k.radii().mean().item():.4f} "
            f"rho_f={model.fast_k.radii().mean().item():.4f}"
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


def predict_features(model, history):
    # history 仅含 10 个历史 patch，不含预测目标 patch。
    slow, fast, _ = model.states(
        history, CONTEXT
    )
    return torch.cat(
        (
            model.slow_k(slow),
            model.fast_k(fast),
        ),
        dim=-1,
    )


def train_and_evaluate_head(args, device):
    train, val, test = make_loaders(
        args.csv,
        CONTEXT + 1,
        args.batch_size,
    )

    saved = torch.load(
        args.ckpt, map_location="cpu"
    )

    if saved["variant"] != args.variant:
        raise ValueError(
            "checkpoint 中的 variant 与命令行参数不同。"
        )

    model = KoopmanTSJEPA(
        variant=saved["variant"],
        dim=saved["dim"],
        slow_dim=saved["slow_dim"],
        fast_dim=saved["fast_dim"],
        local_window=saved["local_window"],
    ).to(device)

    model.load_state_dict(saved["model"])
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    head = nn.Linear(
        saved["slow_dim"] + saved["fast_dim"],
        PATCH,
    ).to(device)

    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.head_lr,
    )

    def evaluate(loader):
        head.eval()
        squared = 0.0
        absolute = 0.0
        count = 0

        with torch.no_grad():
            for patches in loader:
                patches = patches.to(device)

                history = patches[:, :CONTEXT]
                target = patches[:, CONTEXT]

                features = predict_features(
                    model, history
                )

                # 与上一版 B/C 相同的残差预测头。
                pred = (
                    history[:, -1, -1:]
                    + head(features)
                )

                squared += (
                    (pred - target).square().sum().item()
                )
                absolute += (
                    (pred - target).abs().sum().item()
                )
                count += target.numel()

        return squared / count, absolute / count

    best_val = float("inf")
    best_head = None
    best_epoch = None

    for epoch in range(1, args.head_epochs + 1):
        head.train()

        for patches in train:
            patches = patches.to(device)

            history = patches[:, :CONTEXT]
            target = patches[:, CONTEXT]

            with torch.no_grad():
                features = predict_features(
                    model, history
                )

            pred = (
                history[:, -1, -1:]
                + head(features)
            )
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_mse, val_mae = evaluate(val)

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
    test_mse, test_mae = evaluate(test)

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
        choices=["A_prime", "B", "C"],
        required=True,
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--init_encoder", default=""
    )

    parser.add_argument(
        "--dim", type=int, default=128
    )
    parser.add_argument(
        "--slow_dim", type=int, default=32
    )
    parser.add_argument(
        "--fast_dim", type=int, default=32
    )
    parser.add_argument(
        "--local_window", type=int, default=3
    )

    parser.add_argument(
        "--batch_size", type=int, default=64
    )
    parser.add_argument(
        "--epochs", type=int, default=50
    )
    parser.add_argument(
        "--head_epochs", type=int, default=50
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4
    )
    parser.add_argument(
        "--head_lr", type=float, default=1e-3
    )
    parser.add_argument(
        "--ema", type=float, default=0.998
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()
    seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    if args.mode == "pretrain":
        train_pretrain(args, device)
    else:
        train_and_evaluate_head(args, device)


if __name__ == "__main__":
    main()