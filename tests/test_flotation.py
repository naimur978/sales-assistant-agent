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
    assert "Not unusual, but not good either" in render(assess(Readings(silica=3.1, iron=64.5)))


def test_readings_outside_observed_range_are_warned():
    a = assess(Readings(silica=20, iron=40))
    assert not a["position"]["inside_observed_range"]
    assert "WARNING" in render(a)


def test_setting_advice_is_withheld_when_no_variable_is_consistent():
    g = assess(Readings(silica=3.1, iron=64.5))["good_hours"]
    assert g["meaningful"] and g["consistent"] == [] and g["groups_compared"] == 6
    assert "cannot tell us which setting to change" in render(assess(Readings(silica=3.1, iron=64.5)))


def test_episodes_are_counted_from_consecutive_hours():
    df = load()
    e = assess(Readings(silica=3.1, iron=64.5))["episodes"]
    assert e["share_hours"] == pytest.approx((df[SI] >= 3.1).mean())
    assert 1 <= e["median_h"] <= e["mean_h"] <= e["longest_h"] and e["count"] < (df[SI] >= 3.1).sum()
    assert assess(Readings(silica=99, iron=64.5))["episodes"] == {"count": 0}


def test_invalid_readings_rejected():
    for bad in (dict(silica=-1, iron=64), dict(silica=3, iron=140)):
        with pytest.raises(ValidationError):
            Readings(**bad)


def test_the_customers_own_targets_decide_the_verdict():
    out = render(assess(Readings(silica=3.1, iron=64.5, silica_max=2.5, iron_min=65)))
    quick = out.split("Details:")[0]
    assert "- Out of their target. Silica 3.1% is above their 2.5% limit, and iron 64.5% is below their 65% minimum." in quick
    assert "of hours meet that target" in quick
    assert "Within their target." in render(assess(Readings(silica=2.0, iron=66, silica_max=2.5, iron_min=65)))


def test_best_hours_comparison_follows_the_customers_silica():
    a, b = assess(Readings(silica=3.1, iron=64.5))["good_hours"], assess(Readings(silica=4.5, iron=64.5))["good_hours"]
    assert a["meaningful"] and b["meaningful"] and b["n_poor"] < a["n_poor"]
    assert "hours at 4.5% silica or more" in render(assess(Readings(silica=4.5, iron=64.5)))


def test_comparison_is_skipped_when_the_reading_is_not_worse_than_typical():
    a = assess(Readings(silica=1.5, iron=64.5))
    assert not a["good_hours"]["meaningful"] and "too typical" in render(a)
