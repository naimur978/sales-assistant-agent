"""Rough cost estimate for a field service visit, computed in plain code from the visits table.
No LLM here. Nothing is converted between currencies: the reports contain no exchange rates.

Two views are returned:
  1. observed_totals  - what comparable past visits actually cost, per currency as reported
  2. bottom_up        - on-site hours (from similar jobs) x labour rate, travel hours x travel rate
                        (both from the target country's own reports), plus parts priced from similar jobs
"""
from collections import Counter, defaultdict
from statistics import median

from pydantic import BaseModel

from src.ingest.services import EQUIPMENT_LABELS, JOB_LABELS, Equipment, JobTag
from src.slots import canonical_country

JOBS_WITH_QUANTITY = {"cloth_replacement": "cloth", "wear_part_replacement": None}
MIN_SIMILAR = 3


class EstimateRequest(BaseModel):
    equipment: Equipment
    job: JobTag
    country: str | None = None
    quantity: float | None = None  # e.g. 12 cloths


def comparable_total(v: dict) -> float | None:
    """The visit total without VAT when VAT is separated, else the stated total."""
    return v.get("total_ex_vat") or v.get("total_cost")


def part_amount(p: dict) -> float | None:
    """Printed line total if there is one, else quantity x printed unit price."""
    if p.get("line_total") is not None:
        return p["line_total"]
    if p.get("qty") is not None and p.get("unit_price") is not None:
        return p["qty"] * p["unit_price"]
    return None


def derived_total(v: dict) -> float | None:
    """Rebuild a total from printed hours x rates + itemised parts. Only when every piece is printed."""
    need = (v.get("onsite_hours"), v.get("travel_hours"), v.get("labour_rate"), v.get("travel_rate"))
    amounts = [part_amount(p) for p in v.get("parts") or []]
    if None in need or None in amounts:
        return None
    return round(v["onsite_hours"] * v["labour_rate"] + v["travel_hours"] * v["travel_rate"] + sum(amounts), 2)


def _span(values: list[float]) -> dict | None:
    return {"min": min(values), "median": median(values), "max": max(values), "n": len(values)} if values else None


def _same(a: str | None, b: str | None) -> bool:
    return bool(a and b and a.strip().lower() == b.strip().lower())


def estimate(req: EstimateRequest, visits: list[dict]) -> dict:
    questions = []
    if not req.country:
        questions.append("Which country (and ideally which site) is the visit in? Travel time and rates depend on it.")
    if req.job in JOBS_WITH_QUANTITY and req.quantity is None:
        questions.append("How many parts need replacing (for example how many filter cloths)?")
    if questions:
        return {"status": "needs_info", "questions": questions}

    # "chille" or "Chilean" must find the Chile reports, otherwise the labour and travel budget would silently disappear
    req = req.model_copy(update={"country": canonical_country(req.country, sorted({v["country"] for v in visits if v["country"]}))})
    similar = [v for v in visits if v["equipment"] == req.equipment and req.job in v["job_tags"]]
    notes = []
    if len(similar) < MIN_SIMILAR:
        return {"status": "insufficient_data", "similar_ids": [v["report_id"] for v in similar],
                "message": f"Only {len(similar)} past visit(s) match {req.equipment} / {req.job}; "
                           f"not enough to give a range."}

    # 1. observed totals, per currency as reported
    observed = defaultdict(list)
    for v in similar:
        total, kind = comparable_total(v), "stated"
        if total is None:
            total, kind = derived_total(v), "derived from printed hours, rates and parts"
        if total is not None and v["currency"]:
            observed[v["currency"]].append((total, v["report_id"], kind, v["country"]))
    observed_totals = {
        cur: {**_span([t for t, *_ in rows]), "visits": [(rid, t, kind, ctry) for t, rid, kind, ctry in rows]}
        for cur, rows in observed.items()
    }

    # 2. bottom-up for the target country
    local = [v for v in visits if _same(v["country"], req.country)]
    hours = _span([v["onsite_hours"] for v in similar if v["onsite_hours"] is not None])
    travel = _span([v["travel_hours"] for v in local if v["travel_hours"] is not None])
    currencies = Counter(v["currency"] for v in local if v["currency"] and comparable_total(v))
    local_cur = currencies.most_common(1)[0][0] if currencies else None
    labour_rate = _span([v["labour_rate"] for v in local if v["labour_rate"] and v["currency"] == local_cur])
    travel_rate = _span([v["travel_rate"] for v in local if v["travel_rate"] and v["currency"] == local_cur])

    bottom_up = {"currency": local_cur, "onsite_hours": hours, "travel_hours": travel,
                 "labour_rate": labour_rate, "travel_rate": travel_rate, "service_only": None, "parts": {}}
    if not local:
        notes.append(f"There are no past reports from {req.country}, so travel time and rates could not be taken from there.")
    elif not (labour_rate and travel_rate and travel and hours):
        notes.append(f"The reports from {req.country} do not give enough hours and rates for a breakdown.")
    else:
        parts_lo = {"labour": hours["min"] * labour_rate["min"], "travel": travel["min"] * travel_rate["min"]}
        parts_md = {"labour": hours["median"] * labour_rate["median"], "travel": travel["median"] * travel_rate["median"]}
        parts_hi = {"labour": hours["max"] * labour_rate["max"], "travel": travel["max"] * travel_rate["max"]}
        bottom_up["service_only"] = {
            "low": sum(parts_lo.values()), "median": sum(parts_md.values()), "high": sum(parts_hi.values()),
            "labour": {"low": parts_lo["labour"], "median": parts_md["labour"], "high": parts_hi["labour"]},
            "travel": {"low": parts_lo["travel"], "median": parts_md["travel"], "high": parts_hi["travel"]},
        }

    keyword = JOBS_WITH_QUANTITY.get(req.job)
    if keyword and req.quantity:
        prices = defaultdict(list)
        for v in similar:
            for p in v["parts"]:
                if keyword in p["name"].lower() and p.get("unit_price"):
                    prices[v["currency"]].append((p["unit_price"], v["report_id"]))
        for cur, rows in prices.items():
            unit = _span([u for u, _ in rows])
            bottom_up["parts"][cur] = {"unit_price": unit, "ids": [r for _, r in rows],
                                       "low": unit["min"] * req.quantity, "median": unit["median"] * req.quantity,
                                       "high": unit["max"] * req.quantity}
        if not prices:
            notes.append("No past report prints a unit price for these parts, so parts cost is not included.")
        elif local_cur and local_cur not in prices:
            notes.append(f"Part prices are only available in {' and '.join(prices)}, so they are not added to the {local_cur} budget "
                         f"(the reports give no exchange rates).")

    n_stated = sum(1 for v in similar if comparable_total(v) is not None)
    if n_stated < len(similar):
        notes.append(f"{len(similar) - n_stated} of the {len(similar)} similar visits give no total cost, so they only show hours.")
    if req.country and not any(_same(v["country"], req.country) for v in similar):
        notes.append(f"None of the similar visits were in {req.country}, so the totals above include travel and rates from other countries.")

    none_local = bool(req.country) and not any(_same(v["country"], req.country) for v in similar)
    return {"status": "estimate", "none_local": none_local, "request": req.model_dump(), "similar_ids": [v["report_id"] for v in similar],
            "observed_totals": observed_totals, "bottom_up": bottom_up, "notes": notes}


def _money(x: float) -> str:
    return f"{x:,.0f}"


def _to(lo: float, hi: float, fmt=_money) -> str:
    return fmt(lo) if lo == hi else f"{fmt(lo)} to {fmt(hi)}"


def _g(x: float) -> str:
    return f"{x:g}"


def _visits(n: int) -> str:
    return f"{n} visit" + ("" if n == 1 else "s")


def render(result: dict) -> str:
    """Plain-text answer for the salesperson. The budget comes first, then the breakdown, the sources and a few
    plain caveats. Every figure traces to the listed report ids."""
    if result["status"] == "needs_info":
        return "Ask the user (do not guess or retry until they answer):\n" + "\n".join(f"- {q}" for q in result["questions"])
    if result["status"] == "insufficient_data":
        return result["message"]
    r, bu = result["request"], result["bottom_up"]
    out = [f"Estimate for {JOB_LABELS[r['job']]}" + (f" ({r['quantity']:g})" if r["quantity"] else "")
           + f" on {EQUIPMENT_LABELS[r['equipment']]}" + (f" in {r['country']}" if r["country"] else ""), ""]
    so, cur = bu["service_only"], bu["currency"]
    quick = ["Quick answer:"]
    if so:
        quick.append(f"- Budget about {_money(so['median'])} {cur} for labour and travel (range {_money(so['low'])} to {_money(so['high'])}).")
    else:
        quick.append("- No labour and travel budget could be built from the reports, so see the totals of similar visits below.")
    if bu["parts"]:
        extra = " or ".join(f"{_to(p['low'], p['high'])} {pc}" for pc, p in bu["parts"].items())
        quick.append(f"- On top of that, parts cost {extra}, not converted because the reports give no exchange rates.")
    where = f", none in {r['country']}" if result.get("none_local") and r["country"] else ""
    rough = ", so treat this as a rough guide." if result.get("none_local") or result["notes"] else "."
    quick.append(f"- Based on {len(result['similar_ids'])} similar past visits{where}{rough}")
    out += [*quick, "", "Details:", ""]
    if so:
        h, t, lr, tr = bu["onsite_hours"], bu["travel_hours"], bu["labour_rate"], bu["travel_rate"]
        out += [f"Budget for labour and travel ({cur}):",
                f"  Typical    about {_money(so['median'])}",
                f"  Range      {_money(so['low'])} to {_money(so['high'])}", "",
                "Breakdown:",
                f"  Labour     {_to(h['min'], h['max'], _g)} hours on site, at {_to(lr['min'], lr['max'], _g)} {cur} per hour",
                f"  Travel     {_to(t['min'], t['max'], _g)} hours, at {_to(tr['min'], tr['max'], _g)} {cur} per hour", ""]
    if bu["parts"]:
        noun = "cloth" if JOBS_WITH_QUANTITY.get(r["job"]) == "cloth" else "part"
        qty = r["quantity"]
        label = f"{qty:g} {noun}" + ("" if qty == 1 else "s")
        out.append("Parts, not converted:")
        for pc, p in bu["parts"].items():
            u = p["unit_price"]
            out.append(f"  {label} at {_to(u['min'], u['max'], _g)} {pc} each, {_to(p['low'], p['high'])} {pc} in total")
        out.append("")
    out.append("Totals of similar visits, as reported:")
    rebuilt = []
    for c, o in result["observed_totals"].items():
        out.append(f"  {c}   {_to(o['min'], o['max']):<18}{_visits(o['n'])}")
        rebuilt += [rid for rid, _, kind, _ in o["visits"] if "derived" in kind]
    out += ["", "Based on " + f"{len(result['similar_ids'])} similar past visits:", "  " + ", ".join(result["similar_ids"]), ""]
    notes = list(result["notes"])
    if rebuilt:
        notes.append(f"The total for {', '.join(rebuilt)} was rebuilt from its hours, rates and parts, because the report ends before its total.")
    if notes:
        out += ["Good to know:"] + [f"- {n}" for n in notes]
    return "\n".join(out).rstrip()
