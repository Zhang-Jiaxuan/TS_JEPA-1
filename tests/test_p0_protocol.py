# tests/test_p0_protocol.py
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loaders.p0_protocol import load_series_protocol


class P0ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "toy.csv"

        n = 1000
        values = np.arange(n, dtype=np.float32)
        # 验证／测试故意引入极端值：若 scaler 泄漏，train 的统计量会变。
        values[650:] += 100_000

        frame = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=n, freq="h"),
            "OT": values,
        })
        # CSV 原始行打乱，用来检验先排序再划分。
        frame.sample(frac=1, random_state=7).to_csv(
            self.path, index=False
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_only_scaler_and_sorted_split(self):
        p = load_series_protocol(str(self.path))
        self.assertEqual(p.split.train_end, 650)
        self.assertEqual(p.split.val_end, 700)
        self.assertAlmostEqual(
            float(p.scaler.mean[0]), float(np.arange(650).mean()), places=3
        )
        self.assertAlmostEqual(
            float(p.values[:650].mean()), 0.0, places=5
        )
        self.assertTrue(np.all(p.timestamps[1:] >= p.timestamps[:-1]))

    def test_window_cannot_cross_split(self):
        p = load_series_protocol(str(self.path))
        starts = p.window_starts("val", history_length=20, future_length=5)
        self.assertEqual(int(starts[0]), 650)
        self.assertEqual(int(starts[-1]), 675)
        history, future = p.get_window("val", 675, 20, 5)
        self.assertEqual(len(history), 20)
        self.assertEqual(len(future), 5)

        with self.assertRaises(ValueError):
            p.get_window("val", 676, 20, 5)

    def test_inverse_scaling(self):
        p = load_series_protocol(str(self.path))
        restored = p.scaler.inverse_transform(p.values[:10])
        np.testing.assert_allclose(
            restored[:, 0], np.arange(10), atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
