"""Short version first, plain text, no invented sources. No network."""
import pytest

from src.config import FLOTATION_CSV
from src.display import clean_markup, format_reply
from src.guard import check_reply
from src.tools.service_cost import EstimateRequest, estimate, render
from tests.test_service_cost import VISITS


# ---- the cost estimate starts with pointers
def cost_text(country="Chile"):
    return render(estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement", country=country, quantity=12), VISITS))


def test_the_cost_estimate_has_a_quick_answer_then_details():
    out = cost_text().split("\n")
    assert out[0].startswith("Estimate for replacing filter cloths (12)")
    q, d = out.index("Quick answer:"), out.index("Details:")
    assert q < d and 3 <= len([line for line in out[q + 1:d] if line.startswith("- ")]) <= 4


def test_the_quick_answer_carries_the_budget_the_parts_and_the_basis():
    out = cost_text()
    quick = out[out.index("Quick answer:"):out.index("Details:")]
    assert "- Budget about" in quick and "for labour and travel (range" in quick
    assert "- On top of that, parts cost" in quick and "not converted" in quick
    assert "- Based on" in quick and "similar past visits" in quick


def test_the_quick_answer_says_when_the_estimate_is_rough():
    assert "treat this as a rough guide" in cost_text("Chile")


def test_the_quick_answer_says_when_none_of_the_visits_were_in_the_country():
    assert "none in Peru" in cost_text("Peru") or "none in" not in cost_text("Chile")  # Peru has similar visits in the sample data
    out = render(estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement", country="Norway", quantity=12), VISITS))
    assert "none in Norway" in out and "- No labour and travel budget could be built" in out


# ---- flotation starts with pointers too
@pytest.mark.skipif(not FLOTATION_CSV.exists(), reason="flotation CSV not present")
def test_the_flotation_answer_has_three_pointers_before_the_details():
    from src.tools.flotation import Readings, assess
    from src.tools.flotation import render as render_flotation
    out = render_flotation(assess(Readings(silica=3.1, iron=64.5))).split("\n")
    q = out.index("Quick answer:")
    d = out.index("", q)
    pointers = [line for line in out[q + 1:d] if line.startswith("- ")]
    assert out[d + 1] == "Worth asking:" and out.index("Details:") > d
    assert len(pointers) == 3 and pointers[0].startswith("- Not unusual, but not good either") and "of hours" in pointers[0]
    assert "Typical is 2.0% silica and 65.2% iron, against their 3.1% and 64.5%" in pointers[1]
    assert "cannot tell us which setting to change" in pointers[2]


# ---- plain text: no Markdown, no LaTeX
def test_markdown_emphasis_headings_and_code_marks_are_removed():
    assert clean_markup("**CellStation** is an option") == "CellStation is an option"
    assert clean_markup("*(Caveat: one plant.)*") == "(Caveat: one plant.)"
    assert clean_markup("# Title\n* item one\n* item two") == "Title\n- item one\n- item two"
    assert clean_markup("use `product_info` here") == "use product_info here"


def test_latex_maths_becomes_plain_text():
    assert clean_markup("fit ($R^2 = -0.22$ linear, $0.07$ boosting)") == "fit (R2 = -0.22 linear, 0.07 boosting)"
    assert clean_markup("hours with silica ($\\ge 3.1\\%$)") == "hours with silica (>= 3.1%)"


def test_ordinary_text_with_stars_or_dollars_is_left_alone():
    assert clean_markup("Totals 5 * 3 and 12 USD") == "Totals 5 * 3 and 12 USD"
    assert clean_markup("- plain bullet [page 2 of larox-pf-ds.pdf]") == "- plain bullet [page 2 of larox-pf-ds.pdf]"


def test_an_invented_page_reference_is_removed_but_a_real_one_stays():
    out = clean_markup("Silica is high [page 1 of assess_flotation]. See [page 2 of larox-pf-ds.pdf].")
    assert "assess_flotation" not in out and "[page 2 of larox-pf-ds.pdf]" in out and "high." in out


def test_the_whole_pasted_reply_comes_out_as_plain_text():
    reply = ("- **CellStation** is an intelligent solution [page 1 of cellstation.pdf].\n"
             "*(Caveat: The data comes from one plant [page 1 of assess_flotation].)*\n"
             "A model ($R^2 = -0.22$) and ($\\ge 3.1\\%$).")
    out = format_reply(reply)
    assert "*" not in out and "$" not in out and "\\" not in out and "assess_flotation" not in out
    assert "CellStation is an intelligent solution [page 1 of cellstation.pdf]" in out.replace("\n   ", " ").replace("\n  ", " ")


# ---- the reply check rejects invented sources
def test_the_reply_check_rejects_a_source_reference_no_tool_gave():
    tool = "Silica is high. Details follow."
    problems = check_reply("Silica is high [page 1 of assess_flotation].", [tool], "")
    assert any("source references not found" in p and "assess_flotation" in p for p in problems)


def test_the_reply_check_accepts_the_references_the_tool_gave():
    tool = "Fact [page 2 of larox-pf-ds.pdf]."
    assert check_reply("A fact [page 2 of larox-pf-ds.pdf].", [tool], "") == []


def test_a_reply_that_drops_the_details_section_is_rejected():
    tool = "Quick answer:\n- Budget about 1,591 USD.\n\nDetails:\n\nLabour 95 USD per hour"
    assert any("Details" in p for p in check_reply("Quick answer:\n- Budget about 1,591 USD.", [tool], ""))
    assert not check_reply("Quick answer:\n- Budget about 1,591 USD.\n\nDetails:\nLabour 95 USD per hour", [tool], "")
