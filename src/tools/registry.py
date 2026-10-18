"""The three tools the agent can call: name, description the model sees, validated arguments, runner."""
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.config import BROCHURE_DIR
from src.ingest.services import Equipment, JobTag


class ProductArgs(BaseModel):
    question: str = Field(min_length=5, description="Self-contained question about Metso products, e.g. "
                          "'Which Larox PF filter option helps when feed conditions change?'")


class CostArgs(BaseModel):
    equipment: Equipment | None = Field(default=None, description="Null if the user has not named the equipment. Equipment family: larox_pf (Larox PF pressure filter), larox_pf_ds "
                                 "(Larox PF-DS tower press), columncell, concorde_cell, cellstation, filter_optimizer, "
                                 "md_pump, orion_pump, sala_vsp750 (SALA VSP-750 sump pump), or other.")
    job: JobTag | None = Field(default=None, description="Null if the user has not said what work is needed. Type of work: cloth_replacement (filter cloths), wear_part_replacement "
                        "(impellers, rotors, seals, plates), inspection_or_preventive_maintenance, "
                        "software_update_or_calibration, commissioning_or_installation, emergency_repair, cleaning.")
    country: str | None = Field(default=None, description="Country of the customer site. Null if the user has not said.")
    quantity: float | None = Field(default=None, gt=0, description="How many parts are replaced (e.g. 12 filter cloths). "
                                   "Null if the user has not said.")


class FlotationArgs(BaseModel):
    silica: float = Field(ge=0, le=100, description="% silica in the concentrate, as the customer reports")
    iron: float = Field(ge=0, le=100, description="% iron in the concentrate, as the customer reports")


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    run: Callable[[BaseModel], dict]


# What the model sees about each tool. Kept at module level so tests can check the wording without the API.
TOOL_SPECS = {
    "product_info": (
        "Answer a question about Metso products (Larox pressure filters, flotation cells, pumps, filter optimizer, "
        "etc.) using ONLY the product brochures. Every claim comes with a verbatim-verified source (file, page). "
        "Returns status 'answered', 'abstained' (not in the brochures) or 'clarify'. Use for features, specs, "
        "applications and product recommendations: when the customer describes a need, the answer starts with "
        "'Recommendation: <product>' tied to the customer's own words. Not for prices, costs or plant data.",
        ProductArgs),
    "estimate_service_cost": (
        "Estimate the cost of a Metso field service visit from past service reports (hours, travel, rates, parts). "
        "Needs the equipment, the type of job and the country; for parts jobs also the quantity. If something is "
        "missing it returns status 'needs_info' with the question to ask the user. Costs stay in the currency of "
        "the reports; nothing is converted. Not for product features.",
        CostArgs),
    "assess_flotation": (
        "Compare a customer's concentrate silica % and iron % with 4,097 hours of historical flotation plant data: "
        "how common the readings are, what is typical, and what the process data can and cannot say about causes. "
        "Needs both numbers; if one is missing, ask the user for it. Not for product features or service costs.",
        FlotationArgs),
}


def jsonable(obj) -> dict:
    """Make a result safe to send as JSON (numpy numbers, timestamps and tuples become plain values)."""
    return json.loads(json.dumps(obj, default=str))


# Why a claim was dropped, in words that do not repeat the claim itself. The dropped text is never sent back:
# the assistant must not be able to quote a claim that failed verification.
_REASON_KEYS = (("cited page", "cited_page_missing"), ("quote too short", "quote_too_short"),
                ("quote not found", "quote_not_on_page"), ("number ", "number_not_in_quote"),
                ("quote does not fully support", "quote_does_not_support_claim"))


def _reason_key(reason: str) -> str:
    return next((key for prefix, key in _REASON_KEYS if reason.startswith(prefix)), "other")


def product_result(ans) -> dict:
    """Text for the user plus exact fields the assistant can check, and a count of what verification removed."""
    left_out = Counter(_reason_key(why) for _, why in ans.dropped)
    n_out = sum(left_out.values())
    text = ans.text
    if ans.status == "answered" and n_out:
        text += f"\n\n(Note: {n_out} further claim(s) could not be verified and were left out, so this answer may be partial.)"
    data = {"recommended_product": ans.recommended,
            "claims": [{"claim": c.claim, "file": c.file, "page": c.page, "from_table": c.from_table} for c in ans.claims],
            "claims_left_out": n_out, "left_out_reasons": dict(left_out), "closest_pages": ans.closest}
    return {"status": ans.status, "text": text, "data": data}


def cost_result(r: dict) -> dict:
    from src.tools.service_cost import render
    return {"status": r["status"], "text": render(r), "data": jsonable(r)}


def build_tools(context: dict | None = None) -> list[Tool]:
    context = context if context is not None else {}
    from src.grounded import answer
    from src.ingest.brochures import load_chunks
    from src.ingest.services import load_visits
    from src.tools import flotation
    from src.tools.product_search import ProductSearch
    from src.tools.service_cost import EstimateRequest, estimate

    search = ProductSearch(load_chunks(BROCHURE_DIR))

    def product_info(a: ProductArgs) -> dict:
        r = answer(a.question, search, also=context.get("last_user_message", ""))
        return product_result(r)

    def service_cost(a: CostArgs) -> dict:
        ask = []
        if a.equipment is None:
            ask.append("Which equipment is the visit for (for example Larox PF-DS, ColumnCell, Concorde Cell, an MD pump)?")
        if a.job is None:
            ask.append("What work is needed (for example replacing filter cloths, an impeller, a software update, an inspection)?")
        if ask:
            return {"status": "needs_info", "text": "Ask the user (do not guess): " + " ".join(ask), "data": {"missing": ask}}
        return cost_result(estimate(EstimateRequest(**a.model_dump()), load_visits()))

    def assess_flotation(a: FlotationArgs) -> dict:
        result = flotation.assess(flotation.Readings(**a.model_dump()))
        return {"status": "ok", "text": flotation.render(result), "data": jsonable(result)}

    runners = {"product_info": product_info, "estimate_service_cost": service_cost, "assess_flotation": assess_flotation}
    return [Tool(name, description, args, runners[name]) for name, (description, args) in TOOL_SPECS.items()]
