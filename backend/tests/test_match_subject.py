"""🎯 Match colours: the subject takes on the background's colour (and the other way), leaving the rest alone."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import color_match  # noqa: E402


def _scene():
    a = np.zeros((120, 160, 3), np.uint8)
    a[:] = (40, 70, 200)          # a blue scene
    a[30:90, 50:110] = (220, 60, 40)  # a red subject
    m = np.zeros((120, 160), np.float32)
    m[30:90, 50:110] = 1
    return Image.fromarray(a), m


def test_lab_round_trip():
    a = (np.random.default_rng(0).random((20, 20, 3)) * 255).astype(np.uint8)
    assert np.abs(color_match._from_lab(color_match._to_lab(a)).astype(int) - a).max() <= 2


def test_subject_moves_toward_background():
    im, m = _scene()
    out, info = color_match.match_subject(im, m, "subject", strength=1.0, light=1.0)
    o = np.asarray(out).astype(int)
    assert o[60, 80, 2] > 150 and o[60, 80, 0] < 120  # the subject turned bluish
    assert (o[5, 5] == np.asarray(im)[5, 5]).all()  # the background untouched
    assert info["moved"] > 10


def test_background_direction_and_strength_zero():
    im, m = _scene()
    out, _ = color_match.match_subject(im, m, "background", strength=1.0, light=1.0)
    o = np.asarray(out).astype(int)
    assert o[5, 5, 0] > 150 and (o[60, 80] == np.asarray(im)[60, 80]).all()
    same, info = color_match.match_subject(im, m, "subject", strength=0.0)
    assert np.abs(np.asarray(same).astype(int) - np.asarray(im)).max() <= 2 and info["moved"] == 0


def test_reference_picture():
    im, m = _scene()
    ref = Image.new("RGB", (50, 50), (40, 200, 60))
    out, _ = color_match.match_subject(im, m, "subject", strength=1.0, light=1.0, reference=ref)
    assert np.asarray(out)[60, 80, 1] > 150  # greenish now
