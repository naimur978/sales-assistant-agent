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
        notes.append(f"No past reports from {req.country}: travel time and rates could not be taken from that country.")
    elif not (labour_rate and travel_rate and travel and hours):
        notes.append(f"Reports from {req.country} do not give enough hours/rates to build a bottom-up estimate.")
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
            notes.append(f"Part prices are only available in {', '.join(prices)}, not {local_cur}; "
                         f"they are shown separately and NOT converted (the reports contain no exchange rates).")

    n_stated = sum(1 for v in similar if comparable_total(v) is not None)
    if n_stated < len(similar):
        notes.append(f"{len(similar) - n_stated} of {len(similar)} comparable visits do not state a total "
                     f"(e.g. 'as per agreement' / 'invoice to follow'), so they only inform hours, not cost.")
    if req.country and not any(_same(v["country"], req.country) for v in similar):
        notes.append(f"None of the comparable visits were in {req.country}; the observed totals include other countries' "
                     f"travel and rates.")

    return {"status": "estimate", "request": req.model_dump(), "similar_ids": [v["report_id"] for v in similar],
            "observed_totals": observed_totals, "bottom_up": bottom_up, "notes": notes}


def render(result: dict) -> str:
    """Plain-text summary for the salesperson. Every figure traces to listed report IDs."""
    if result["status"] == "needs_info":
        return "Ask the user (do not guess or retry until they answer):\n" + "\n".join(f"- {q}" for q in result["questions"])
    if result["status"] == "insufficient_data":
        return result["message"]
    r, bu = result["request"], result["bottom_up"]
    out = [f"Estimate for {JOB_LABELS[r['job']]}" + (f" ({r['quantity']:g})" if r["quantity"] else "")
           + f" on {EQUIPMENT_LABELS[r['equipment']]}" + (f" in {r['country']}" if r["country"] else "")
           + f", based on {len(result['similar_ids'])} similar past visits ({', '.join(result['similar_ids'])}).", ""]
    so, cur = bu["service_only"], bu["currency"]
    if so:
        h, t = bu["onsite_hours"], bu["travel_hours"]
        out += [f"Labour + travel ({cur}, {r['country']} rates): about {so['median']:,.0f} typical "
                f"(range {so['low']:,.0f}-{so['high']:,.0f}).",
                f"  labour: {h['min']:g}-{h['max']:g} h on site (median {h['median']:g}, n={h['n']}) at "
                f"{bu['labour_rate']['min']:g}-{bu['labour_rate']['max']:g} {cur}/h",
                f"  travel: {t['min']:g}-{t['max']:g} h (median {t['median']:g}, n={t['n']}) at "
                f"{bu['travel_rate']['min']:g}-{bu['travel_rate']['max']:g} {cur}/h"]
    for pc, p in bu["parts"].items():
        u = p["unit_price"]
        out.append(f"Parts ({pc}, not converted): {r['quantity']:g} x {u['min']:g}-{u['max']:g} per unit = "
                   f"{p['low']:,.0f}-{p['high']:,.0f} {pc} (from {', '.join(p['ids'])}).")
    out.append("")
    out.append("What similar visits actually cost (as reported, all-in, not converted):")
    for c, o in result["observed_totals"].items():
        out.append(f"  {c}: {o['min']:,.0f}-{o['max']:,.0f} (median {o['median']:,.0f}, n={o['n']}) - "
                   + ", ".join(f"{rid}{'*' if 'derived' in kind else ''}" for rid, _, kind, _ in o["visits"]))
    if any("derived" in k for o in result["observed_totals"].values() for _, _, k, _ in o["visits"]):
        out.append("  * total rebuilt from printed hours x rates + parts because the report's own total is missing")
    out += [""] + [f"Note: {n}" for n in result["notes"]]
    return "\n".join(out)
