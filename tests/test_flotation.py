import pytest
from pydantic import ValidationError

from src.config import FLOTATION_CSV
from src.tools.flotation import FE, SI, Readings, assess, load, render

pytestmark = pytest.mark.skipif(not FLOTATION_CSV.exists(), reason="flotation CSV not present")


def test_position_matches_direct_pandas_counts():
    df = load()
    p = assess(Readings(silica=3.1, iron=64.5))["position"]
    assert p["silica_share_at_or_above"] == pytest.approx((df[SI] >= 3.1).mean())
    assert p["iron_share_at_or_below"] == pytest.approx((df[FE] <= 64.5).mean())
    assert p["both_share"] == pytest.approx(((df[SI] >= 3.1) & (df[FE] <= 64.5)).mean())
    assert p["n_hours"] == 4097


def test_typical_readings_are_not_called_worse_than_typical():
    text = render(assess(Readings(silica=1.5, iron=66.5)))
    assert "typical or better" in text


def test_poor_readings_are_flagged_on_both():
    assert "worse than this plant's typical performance on both" in render(assess(Readings(silica=3.1, iron=64.5)))


def test_readings_outside_observed_range_are_warned():
    a = assess(Readings(silica=20, iron=40))
    assert not a["position"]["inside_observed_range"]
    assert "WARNING" in render(a)


def test_causal_advice_is_withheld_when_model_does_not_generalise():
    d = assess(Readings(silica=3.1, iron=64.5))["drivers"]
    assert not d["trusted"] and max(d["test_r2_linear"], d["test_r2_gbm"]) < 0.3
    assert "do NOT support firm advice" in render(assess(Readings(silica=3.1, iron=64.5)))


def test_invalid_readings_rejected():
    for bad in (dict(silica=-1, iron=64), dict(silica=3, iron=140)):
        with pytest.raises(ValidationError):
            Readings(**bad)
