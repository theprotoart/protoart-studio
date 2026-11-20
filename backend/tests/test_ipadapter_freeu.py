"""🖼 Image prompt: style-only / layout-only for SDXL (InstantStyle's blocks); ✨ FreeU's settings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine  # noqa: E402


def test_ip_scale_parts():
    assert engine.ip_scale("sdxl", 0.8, "style") == {"up": {"block_0": [0.0, 0.8, 0.0]}}
    assert engine.ip_scale("sdxl", 0.5, "layout") == {"down": {"block_2": [0.0, 0.5]}}
    assert engine.ip_scale("sdxl", 0.6, "all") == 0.6
    assert engine.ip_scale("sd15", 0.6, "style") == 0.6  # SD 1.5: the whole picture


def test_freeu_values():
    assert set(engine.FREEU) == {"sd15", "sdxl"}
    for v in engine.FREEU.values():
        assert set(v) == {"s1", "s2", "b1", "b2"}
