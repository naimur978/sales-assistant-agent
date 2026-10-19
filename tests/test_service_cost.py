"""Tests for the visits table (checked against facts read by hand from the reports) and the estimator."""
import json
from pathlib import Path

import pytest

from src.ingest.services import TABLE, Visit, load_visits, number_values, validate
from src.tools.service_cost import EstimateRequest, comparable_total, derived_total, estimate

needs_table = pytest.mark.skipif(not TABLE.exists(), reason="run `python -m src.ingest.services` first")

# Values I read directly from the report text (tests/service_golden.json). None = the report does not state it.
GOLDEN = json.loads((Path(__file__).parent / "service_golden.json").read_text())


@needs_table
@pytest.mark.parametrize("rid", GOLDEN)
def test_extraction_matches_hand_read_values(rid):
    row = {r["report_id"]: r for r in load_visits()}[rid]
    for field, want in GOLDEN[rid].items():
        got = comparable_total(row) if field == "total" else row[field]
        assert got == want, f"{rid}.{field}: extracted {got!r}, report says {want!r}"


@needs_table
def test_every_visit_has_core_fields():
    for r in load_visits():
        assert r["equipment"] != "" and r["job_tags"], r["report_id"]
    assert len(load_visits()) == 50


@needs_table
def test_rebuilt_totals_reproduce_stated_totals():
    """Where a report prints hours, rates, parts AND a total, hours x rate + parts must equal the total.
    This is what justifies using the rebuilt total for the truncated FSR-0027."""
    checked = 0
    for r in load_visits():
        stated, rebuilt = comparable_total(r), derived_total(r)
        if stated and rebuilt:
            checked += 1
            assert abs(stated - rebuilt) / stated < 0.02, f"{r['report_id']}: stated {stated}, rebuilt {rebuilt}"
    assert checked >= 5


def test_validation_nulls_numbers_not_in_the_report():
    text = "Total on-site hours: 7.4\nTravel: 5.6 hours\nTotal cost: $2,150"
    v = Visit(equipment="cellstation", job_tags=["software_update_or_calibration"], onsite_hours=7.4,
              travel_hours=9.9, total_cost=2150.0, currency="USD")
    v, issues = validate(v, text)
    assert v.onsite_hours == 7.4 and v.total_cost == 2150.0
    assert v.travel_hours is None and any("travel_hours" in i for i in issues)


def test_number_values_handles_separators():
    assert {2540.0, 3060.0, 1.6, 17690.0} <= number_values("$2,540 and 3,060.00 and 1,6 m and 17,690")


# ---- estimator on a tiny synthetic table ----
def visit(rid, country, onsite, travel, lab, trv, total, qty_price=None, equipment="larox_pf_ds", cur="USD"):
    parts = [{"name": "PF-DS filter cloth", "qty": 12, "unit_price": qty_price}] if qty_price else []
    return dict(report_id=rid, country=country, equipment=equipment, job_tags=["cloth_replacement"],
                onsite_hours=onsite, travel_hours=travel, labour_rate=lab, travel_rate=trv, currency=cur,
                total_cost=total, total_ex_vat=None, parts=parts)


VISITS = [
    visit("A", "Chile", 8, 4, 100, 50, 1000), visit("B", "Chile", 10, 6, 120, 70, 1620),
    visit("C", "Peru", 6, 2, None, None, None), visit("D", "Peru", 12, 8, None, None, 3000, qty_price=40, cur="EUR"),
]


def test_asks_for_country_and_quantity_instead_of_guessing():
    r = estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement"), VISITS)
    assert r["status"] == "needs_info" and len(r["questions"]) == 2


def test_refuses_when_too_few_similar_visits():
    r = estimate(EstimateRequest(equipment="columncell", job="cloth_replacement", country="Chile", quantity=1), VISITS)
    assert r["status"] == "insufficient_data"


def test_bottom_up_uses_target_country_rates_and_travel():
    r = estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement", country="Chile", quantity=12), VISITS)
    so = r["bottom_up"]["service_only"]
    assert so["low"] == 6 * 100 + 4 * 50  # min hours x min labour rate + min travel x min travel rate
    assert so["high"] == 12 * 120 + 6 * 70
    assert r["bottom_up"]["currency"] == "USD"


def test_parts_in_other_currency_are_not_converted():
    r = estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement", country="Chile", quantity=12), VISITS)
    assert "EUR" in r["bottom_up"]["parts"] and r["bottom_up"]["parts"]["EUR"]["low"] == 480
    assert any("not added to the USD budget" in n for n in r["notes"])


def test_unknown_country_gives_no_invented_rates():
    r = estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement", country="Norway", quantity=12), VISITS)
    assert r["bottom_up"]["service_only"] is None
    assert any("Norway" in n for n in r["notes"])


def test_visits_without_totals_are_flagged_not_counted_as_cost():
    r = estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement", country="Chile", quantity=12), VISITS)
    assert sum(o["n"] for o in r["observed_totals"].values()) == 3  # C has no total
    assert any("give no total cost" in n for n in r["notes"])



# ---- how the estimate reads
def text_for(visits=VISITS, **kw):
    from src.tools.service_cost import render
    req = dict(equipment="larox_pf_ds", job="cloth_replacement", country="Chile", quantity=12)
    return render(estimate(EstimateRequest(**{**req, **kw}), visits))


def test_the_budget_comes_first_then_the_breakdown_then_the_sources():
    out = text_for().split("\n")
    assert out[0].startswith("Estimate for replacing filter cloths (12)")
    order = [out.index(h) for h in ("Budget for labour and travel (USD):", "Breakdown:", "Parts, not converted:",
                                    "Totals of similar visits, as reported:")]
    assert order == sorted(order)
    assert any(line.startswith("Based on ") and "similar past visits" in line for line in out)


def test_ranges_use_to_and_a_single_value_is_not_written_as_a_range():
    out = text_for()
    assert " to " in out and "n=" not in out and "median" not in out
    assert "98-98" not in out and not any(f"{x}-{x}" in out for x in ("98", "4,367"))


def test_visits_are_counted_in_words():
    out = text_for()
    assert "1 visit" in out and "visits" in out


def test_caveats_are_plain_sentences_under_good_to_know():
    out = text_for()
    assert "Good to know:" in out and "- Part prices are only available in EUR, so they are not added to the USD budget" in out
    assert "e.g." not in out and "bottom-up" not in out and "'as per agreement'" not in out


def test_a_rebuilt_total_is_explained_in_plain_words():
    from src.tools.service_cost import render
    visits = [dict(v) for v in VISITS]
    visits[1] = {**visits[1], "total_cost": None, "labour_rate": 100, "travel_rate": 50}  # B has hours, rates, no total
    out = render(estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement", country="Chile", quantity=12), visits))
    assert "was rebuilt from its hours, rates and parts" in out



# ---- the country as typed must still find the reports
def test_a_misspelled_or_adjective_country_finds_the_same_reports_as_the_exact_name():
    exact = estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement", country="Chile", quantity=12), VISITS)
    for typed in ("chille", "Chilean", "CHILE", "Chile "):
        got = estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement", country=typed, quantity=12), VISITS)
        assert got["bottom_up"]["service_only"] == exact["bottom_up"]["service_only"], typed
        assert got["request"]["country"] == "Chile"  # shown the way the data spells it


def test_an_unknown_country_stays_as_typed_and_gets_no_invented_rates():
    got = estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement", country="Norway", quantity=12), VISITS)
    assert got["request"]["country"] == "Norway" and got["bottom_up"]["service_only"] is None
