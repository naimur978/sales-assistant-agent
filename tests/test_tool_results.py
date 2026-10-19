"""Tool results the assistant can check: exact fields, sources, and what verification removed. No network."""
import json

from pydantic import BaseModel

from src.agent import Agent
from src.grounded import Answer, ExtractedClaim
from src.tools.registry import Tool, cost_result, jsonable, product_result
from src.tools.service_cost import EstimateRequest, estimate
from tests.test_agent import call, fake_llm, say
from tests.test_service_cost import VISITS

CLAIM = ExtractedClaim(claim="The optimizer keeps performance up.", quote="Optimized filter performance", file="filter-optimizer", page=2)


def answered(dropped=()):
    return Answer("answered", "- The optimizer keeps performance up. [page 2 of filter-optimizer.pdf]", claims=[CLAIM],
                  recommended="Filter optimizer", dropped=list(dropped), closest=["filter-optimizer:2"])


def test_product_result_has_exact_fields_and_sources():
    r = product_result(answered())
    assert r["data"]["claims"][0] == {"claim": CLAIM.claim, "file": "filter-optimizer", "page": 2, "from_table": False}
    assert r["data"]["recommended_product"] == "Filter optimizer" and r["data"]["claims_left_out"] == 0
    assert "Note:" not in r["text"]


def test_left_out_claims_are_counted_by_reason_and_flagged_as_partial():
    dropped = [("secret claim A", "quote not found verbatim on the cited page"),
               ("secret claim B", "number 60 in claim is not in the quote"),
               ("secret claim C", "quote does not fully support claim")]
    r = product_result(answered(dropped))
    assert r["data"]["claims_left_out"] == 3
    assert r["data"]["left_out_reasons"] == {"quote_not_on_page": 1, "number_not_in_quote": 1, "quote_does_not_support_claim": 1}
    assert "the brochures may say more than is shown here" in r["text"] and "Check the pages listed above" in r["text"]
    assert not any(ch.isdigit() for ch in r["text"].split("(Note:")[1])  # no counts for the salesperson to puzzle over


def test_text_of_a_removed_claim_never_reaches_the_assistant():
    r = product_result(answered([("secret claim A", "quote not found verbatim on the cited page")]))
    assert "secret claim A" not in json.dumps(r)


def test_abstained_answer_reports_the_counts_without_adding_a_note():
    r = product_result(Answer("abstained", "not in the brochures", dropped=[("x", "quote not found verbatim")]))
    assert r["data"]["claims_left_out"] == 1 and "Note:" not in r["text"]


def test_cost_result_carries_figures_and_report_ids_as_plain_json():
    r = cost_result(estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement", country="Chile", quantity=12), VISITS))
    json.dumps(r)  # must be serialisable as is
    d = r["data"]
    assert d["status"] == "estimate" and d["similar_ids"] and d["bottom_up"]["service_only"]["median"] > 0
    assert d["bottom_up"]["onsite_hours"]["n"] >= 1 and "EUR" in d["bottom_up"]["parts"]


def test_jsonable_converts_numpy_and_timestamps():
    import numpy as np
    import pandas as pd
    out = jsonable({"a": np.float64(1.5), "b": pd.Timestamp("2017-03-10"), "c": (1, 2), "d": np.bool_(True)})
    assert out["a"] == 1.5 and out["c"] == [1, 2] and isinstance(out["b"], str)


class A(BaseModel):
    text: str


def data_tool():
    return Tool("t", "t", A, lambda a: {"status": "ok", "text": "short text", "data": {"median": 1591.37}})


def test_guard_accepts_a_figure_that_is_only_in_the_data_field():
    a = Agent(tools=[data_tool()], llm=fake_llm([call("t", text="x"), say("The median is about 1,591.37.")]), log=False)
    assert a.ask("go") == "The median is about 1,591.37."


def test_guard_still_rejects_a_figure_that_is_in_neither():
    a = Agent(tools=[data_tool()], llm=fake_llm([call("t", text="x"), say("It is 9,999."), say("Still 8,888.")]), log=False)
    reply = a.ask("go")
    assert "9,999" not in reply and "8,888" not in reply


def test_fallback_shows_the_readable_text_not_the_raw_data():
    a = Agent(tools=[data_tool()], llm=fake_llm([call("t", text="x"), say("It is 9,999."), say("Still 8,888.")]), log=False)
    reply = a.ask("go")
    assert "short text" in reply and "1591.37" not in reply and "{" not in reply
