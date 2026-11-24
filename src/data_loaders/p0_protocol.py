# src/data_loaders/p0_protocol.py
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitSpec:
    train_end: int
    val_end: int
    total: int

    def bounds(self, split: str) -> tuple[int, int]:
        if split == "train":
            return 0, self.train_end
        if split == "val":
            return self.train_end, self.val_end
        if split == "test":
            return self.val_end, self.total
        raise ValueError(f"Unknown split: {split}")


@dataclass(frozen=True)
class TrainScaler:
    columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return values * self.std + self.mean

    def state_dict(self) -> dict:
        return {
            "columns": list(self.columns),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


@dataclass(frozen=True)
class SeriesProtocol:
    # values 已排序、已用训练集统计量标准化；仍保留全序列索引。
    values: np.ndarray
    timestamps: np.ndarray
    split: SplitSpec
    scaler: TrainScaler

    def window_starts(
        self,
        split_name: str,
        history_length: int,
        future_length: int = 0,
        stride: int = 1,
    ) -> np.ndarray:
        if history_length <= 0 or future_length < 0 or stride <= 0:
            raise ValueError("Invalid window length or stride")

        left, right = self.split.bounds(split_name)
        last_start = right - history_length - future_length
        if last_start < left:
            raise ValueError(
                f"{split_name} too short: rows={right-left}, "
                f"required={history_length+future_length}"
            )

        return np.arange(left, last_start + 1, stride, dtype=np.int64)

    def get_window(
        self, split_name: str, start: int,
        history_length: int, future_length: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        left, right = self.split.bounds(split_name)
        end = start + history_length + future_length
        if start < left or end > right:
            raise ValueError(
                f"Window [{start}, {end}) crosses {split_name} bounds "
                f"[{left}, {right})"
            )
        history_end = start + history_length
        return (
            self.values[start:history_end],
            self.values[history_end:end],
        )


def load_series_protocol(
    path: str,
    columns: tuple[str, ...] = ("OT",),
    timestamp_col: str = "date",
    val_fraction: float = 0.05,
    test_fraction: float = 0.30,
) -> SeriesProtocol:
    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation/test fractions must be positive")
    if val_fraction + test_fraction >= 1:
        raise ValueError("No rows would remain for training")

    df = pd.read_csv(Path(path))
    required = {timestamp_col, *columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="raise")
    df = df.sort_values(timestamp_col, kind="stable").reset_index(drop=True)

    raw = df.loc[:, list(columns)].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Input contains NaN or Inf")

    total = len(df)
    val_len = int(total * val_fraction)
    test_len = int(total * test_fraction)
    train_end = total - val_len - test_len
    val_end = total - test_len
    if min(train_end, val_len, test_len) <= 1:
        raise ValueError("At least two rows are required in each split")

    # 与 pandas Series.std() 一样使用样本标准差 ddof=1。
    mean = raw[:train_end].mean(axis=0)
    std = raw[:train_end].std(axis=0, ddof=1)
    # 常量列不能被除零；标准差置 1 后该列标准化为 0。
    std = np.where(std < 1e-8, 1.0, std)

    scaler = TrainScaler(
        columns=columns,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )
    return SeriesProtocol(
        values=scaler.transform(raw),
        timestamps=df[timestamp_col].to_numpy(),
        split=SplitSpec(train_end, val_end, total),
        scaler=scaler,
    )
