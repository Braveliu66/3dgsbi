from __future__ import annotations

import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
METRICS_SOURCE = WORKSPACE_ROOT / "worker" / "trainer" / "dash_deblur_group_gs" / "metrics.py"


class DashDeblurGroupMetricsTests(unittest.TestCase):
    def test_upstream_metric_code_is_unmodified(self) -> None:
        source = METRICS_SOURCE.read_text(encoding="utf-8")

        self.assertIn("multichannel=True", source)
        self.assertNotIn("channel_axis=-1", source)


if __name__ == "__main__":
    unittest.main()
