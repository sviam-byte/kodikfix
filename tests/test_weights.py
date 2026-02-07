import pandas as pd

from src.weights import WeightPolicy, apply_weight_policy_to_series


def test_weight_policy_drop_nonpositive():
    """Drop non-positive and non-finite weights."""
    w = pd.Series([1, 0, -2, 3, None])
    pol = WeightPolicy(mode="drop_nonpositive", eps=1e-9)
    w2, keep = apply_weight_policy_to_series(w, pol)
    kept = w2[keep].tolist()
    assert kept == [1.0, 3.0]


def test_weight_policy_abs():
    """abs policy keeps magnitudes and drops zeros."""
    w = pd.Series([-2, 0, 5])
    pol = WeightPolicy(mode="abs", eps=1e-9)
    w2, keep = apply_weight_policy_to_series(w, pol)
    assert w2[keep].tolist() == [2.0, 5.0]


def test_weight_policy_clip_keeps_finite():
    """clip policy should keep finite values and apply eps floor."""
    w = pd.Series([-2, 0, 5])
    pol = WeightPolicy(mode="clip", eps=0.1)
    w2, keep = apply_weight_policy_to_series(w, pol)
    assert keep.all()
    assert w2.tolist() == [0.1, 0.1, 5.0]


def test_weight_policy_shift_adds_offset():
    """shift policy should add constant offset and keep finite values."""
    w = pd.Series([-0.5, 0.1])
    pol = WeightPolicy(mode="shift", eps=0.1, shift=1.0)
    w2, keep = apply_weight_policy_to_series(w, pol)
    assert keep.all()
    assert w2.tolist() == [0.5, 1.1]
