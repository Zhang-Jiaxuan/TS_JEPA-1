# src/data_loaders/data_class.py

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .p0_protocol import SeriesProtocol, load_series_protocol


def _get_protocol(
    path_data: str | None,
    protocol: SeriesProtocol | None,
    val_fraction: float,
    test_fraction: float,
) -> SeriesProtocol:
    if protocol is not None:
        return protocol
    if path_data is None:
        raise ValueError("Provide either path_data or protocol")
    return load_series_protocol(
        path_data,
        columns=("OT",),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )


def _split_tensors(protocol: SeriesProtocol):
    values = torch.as_tensor(
        np.asarray(protocol.values).reshape(-1),
        dtype=torch.float32,
    )
    train_end = protocol.split.train_end
    val_end = protocol.split.val_end
    return (
        values[:train_end],
        values[train_end:val_end],
        values[val_end:],
    )


class CSVDataLoader(Dataset):
    """
    原 TS-JEPA 随机 Patch mask 数据集。

    每个样本返回：
        patches:          [num_patches, patch_size]
        mask_indices:     [num_masked]
        non_mask_indices: [num_visible]

    注意：mask_indices 保留 random.sample 的顺序，与原实现一致。
    """

    def __init__(
        self,
        path_data: str | None = None,
        batch_size: int = 32,
        series_split_size: int = 320,
        patch_size: int = 32,
        mask_ratio: float = 0.15,
        split: str = "train",
        protocol: SeriesProtocol | None = None,
        val_fraction: float = 0.05,
        test_fraction: float = 0.30,
    ):
        super().__init__()

        if patch_size <= 0 or series_split_size <= 0:
            raise ValueError("patch_size and series_split_size must be positive")
        if series_split_size % patch_size != 0:
            raise ValueError(
                "series_split_size must be divisible by patch_size"
            )
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must lie strictly between 0 and 1")

        self.protocol = _get_protocol(
            path_data, protocol, val_fraction, test_fraction
        )
        self.split = split
        self.batch_size = batch_size  # 兼容原构造参数
        self.series_split_size = series_split_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = series_split_size // patch_size

        if self.num_patches < 2:
            raise ValueError("Random masking requires at least two patches")

        self.train_df, self.val_df, self.test_df = _split_tensors(
            self.protocol
        )

        split_series = {
            "train": self.train_df,
            "val": self.val_df,
            "test": self.test_df,
        }
        if split not in split_series:
            raise ValueError(f"Unknown split: {split}")

        self.time_series_list = split_series[split]

        # 与原实现一致：以不重叠的完整窗口取样，舍弃 split 尾部不足
        # series_split_size 的数据。没有跨 split 取上下文。
        self.starts = self.protocol.window_starts(
            split,
            history_length=series_split_size,
            future_length=0,
            stride=series_split_size,
        )

        n_masked = int(self.num_patches * mask_ratio)
        if not 1 <= n_masked < self.num_patches:
            raise ValueError(
                f"mask_ratio={mask_ratio} gives {n_masked} masked patches "
                f"out of {self.num_patches}; need at least one masked "
                "and one visible patch"
            )
        self.num_masked = n_masked

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])

        history, _ = self.protocol.get_window(
            self.split,
            start,
            history_length=self.series_split_size,
        )
        patches = torch.as_tensor(
            np.asarray(history).reshape(
                self.num_patches, self.patch_size
            ).copy(),
            dtype=torch.float32,
        )

        # 不能在 P0 阶段排序：原 Predictor 和 teacher 的目标索引
        # 必须继续按原样配对。
        mask_indices_list = random.sample(
            range(self.num_patches), self.num_masked
        )
        mask_set = set(mask_indices_list)
        non_mask_indices_list = [
            i for i in range(self.num_patches) if i not in mask_set
        ]

        mask_indices = torch.tensor(
            mask_indices_list, dtype=torch.long
        )
        non_mask_indices = torch.tensor(
            non_mask_indices_list, dtype=torch.long
        )

        return patches, mask_indices, non_mask_indices


class EvaluationDataLoader(Dataset):
    """
    原预测下游任务：连续 context_size 个 Patch -> 下一个 Patch。

    返回：
        context_patches: [context_size, patch_size]
        target_patch:    [patch_size]
    """

    def __init__(
        self,
        path_data: str | None = None,
        patch_size: int = 32,
        context_size: int = 10,
        split: str = "train",
        protocol: SeriesProtocol | None = None,
        val_fraction: float = 0.05,
        test_fraction: float = 0.30,
    ):
        super().__init__()

        if patch_size <= 0 or context_size <= 0:
            raise ValueError("patch_size and context_size must be positive")

        self.protocol = _get_protocol(
            path_data, protocol, val_fraction, test_fraction
        )
        self.patch_size = patch_size
        self.context_size = context_size
        self.split = split

        self.train_df, self.val_df, self.test_df = _split_tensors(
            self.protocol
        )
        split_series = {
            "train": self.train_df,
            "val": self.val_df,
            "test": self.test_df,
        }
        if split not in split_series:
            raise ValueError(f"Unknown split: {split}")

        self.series = split_series[split]

        # 兼容旧脚本可能访问的属性。这里只截取本 split 内完整 Patch。
        n_complete = len(self.series) // patch_size
        self.patches_tensor = self.series[
            : n_complete * patch_size
        ].reshape(n_complete, patch_size)

        total_length = (context_size + 1) * patch_size

        # stride=patch_size：与原代码「窗口每次前进一个 Patch」
        # 的样本定义一致；每个 target 都留在指定 split 内。
        self.starts = self.protocol.window_starts(
            split,
            history_length=context_size * patch_size,
            future_length=patch_size,
            stride=patch_size,
        )

        if total_length > len(self.series):
            raise ValueError(
                f"{split} split has {len(self.series)} points; "
                f"need at least {total_length}"
            )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])

        context, target = self.protocol.get_window(
            self.split,
            start,
            history_length=self.context_size * self.patch_size,
            future_length=self.patch_size,
        )

        context_patches = torch.as_tensor(
            np.asarray(context).reshape(
                self.context_size, self.patch_size
            ).copy(),
            dtype=torch.float32,
        )
        target_patch = torch.as_tensor(
            np.asarray(target).reshape(self.patch_size).copy(),
            dtype=torch.float32,
        )
        return context_patches, target_patch
