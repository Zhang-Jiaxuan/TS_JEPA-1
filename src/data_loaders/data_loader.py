# src/data_loaders/data_loader.py

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_class import CSVDataLoader, EvaluationDataLoader
from .p0_protocol import SeriesProtocol, load_series_protocol


def _seed_worker(worker_id: int):
    # torch.initial_seed 由 DataLoader 的 generator 决定。
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _build_protocol(
    path: str,
    protocol: SeriesProtocol | None,
    val_fraction: float,
    test_fraction: float,
) -> SeriesProtocol:
    if protocol is not None:
        return protocol
    return load_series_protocol(
        path,
        columns=("OT",),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )


def get_jepa_loaders(
    path,
    batch_size,
    ratio_patches=10,
    mask_ratio=0.9,
    *,
    split="train",
    protocol=None,
    patch_size=32,
    seed=42,
    num_workers=0,
    val_fraction=0.05,
    test_fraction=0.30,
):
    """
    返回单个 DataLoader，与原 pretrain.py 调用方式兼容。

    ratio_patches 现在真实决定每个预训练窗口的 Patch 数。
    旧版本忽略此参数并固定使用 320/32=10 个 Patch。
    """
    if ratio_patches < 2:
        raise ValueError("ratio_patches must be at least 2")

    protocol = _build_protocol(
        path, protocol, val_fraction, test_fraction
    )
    dataset = CSVDataLoader(
        path_data=path,
        batch_size=batch_size,
        series_split_size=ratio_patches * patch_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        split=split,
        protocol=protocol,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def get_evaluation_loaders(
    path,
    batch_size,
    ratio_patches=10,
    mask_ratio=0.9,
    *,
    split="train",
    protocol=None,
    patch_size=32,
    seed=42,
    num_workers=0,
    val_fraction=0.05,
    test_fraction=0.30,
):
    """
    返回指定 split 的预测 DataLoader。

    mask_ratio 仅为兼容旧函数签名保留；预测评估不使用 mask。
    """
    del mask_ratio

    protocol = _build_protocol(
        path, protocol, val_fraction, test_fraction
    )
    dataset = EvaluationDataLoader(
        path_data=path,
        patch_size=patch_size,
        context_size=ratio_patches,
        split=split,
        protocol=protocol,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
    )
