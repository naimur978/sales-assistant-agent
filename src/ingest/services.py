"""Turn the field service reports into one table, once.
LLM reads each report (layouts vary a lot); plain code then checks every extracted number
against the report text and nulls anything it cannot find. Output is cached, not re-run.

  python -m src.ingest.services            # build .cache/service_visits.json (+ .csv)
"""
import csv
import json
import logging
import re
from typing import Literal

from pydantic import BaseModel

from src import llm
from src.config import CACHE_DIR, DATA_DIR
from src.prompts import SERVICE_EXTRACT_SYSTEM

REPORT_DIR = DATA_DIR / "HistoricalServices"
TABLE = CACHE_DIR / "service_visits.json"
PARTIAL = CACHE_DIR / "service_visits.partial.jsonl"
log = logging.getLogger(__name__)

Equipment = Literal[
    "columncell", "concorde_cell", "cellstation", "filter_optimizer",
    "larox_pf", "larox_pf_ds", "md_pump", "orion_pump", "sala_vsp750", "other",
]
JobTag = Literal[
    "cloth_replacement", "wear_part_replacement", "inspection_or_preventive_maintenance",
    "software_update_or_calibration", "commissioning_or_installation", "emergency_repair", "cleaning",
]


# Plain wording for replies. The internal names above are for the code, not for the salesperson.
EQUIPMENT_LABELS = {
    "columncell": "a ColumnCell flotation column", "concorde_cell": "a Concorde Cell", "cellstation": "a CellStation control system",
    "filter_optimizer": "a Filter optimizer control system", "larox_pf": "a Larox PF pressure filter",
    "larox_pf_ds": "a Larox PF-DS tower press filter", "md_pump": "an MD series mill discharge pump",
    "orion_pump": "an Orion series slurry pump", "sala_vsp750": "a SALA VSP-750 sump pump", "other": "the equipment",
}
JOB_LABELS = {
    "cloth_replacement": "replacing filter cloths", "wear_part_replacement": "replacing wear parts",
    "inspection_or_preventive_maintenance": "inspection and preventive maintenance",
    "software_update_or_calibration": "a software update or calibration", "commissioning_or_installation": "commissioning or installation",
    "emergency_repair": "an emergency repair", "cleaning": "cleaning",
}


class Part(BaseModel):
    name: str
    qty: float | None = None
    unit_price: float | None = None  # only if the report prints a price per unit
    line_total: float | None = None  # the amount printed for this line (cost table or parts table)


class Visit(BaseModel):
    customer: str | None = None
    site: str | None = None
    country: str | None = None
    year: int | None = None
    equipment: Equipment
    job_tags: list[JobTag]
    quantity_replaced: float | None = None  # e.g. 12 for "12 filter cloths replaced"
    quantity_unit: str | None = None
    onsite_hours: float | None = None
    travel_hours: float | None = None  # total travel, as the report states it
    currency: Literal["USD", "EUR", "CAD", "AUD", "SEK", "ZAR"] | None = None
    currency_explicit: bool = False  # false when only a bare "$" appears
    total_cost: float | None = None  # the total the report states, else null
    total_ex_vat: float | None = None  # only when the report separates VAT
    vat: Literal["included", "excluded", "not_mentioned"] = "not_mentioned"
    labour_rate: float | None = None  # per hour, only if printed
    travel_rate: float | None = None  # per hour, only if printed
    parts: list[Part] = []
    cost_text: str | None = None  # the exact words/line giving the total or saying it is not given


SYSTEM = SERVICE_EXTRACT_SYSTEM


def _squash(text: str) -> str:
    return re.sub(r"[\s*|]+", " ", text).strip().lower()


def number_values(text: str) -> set[float]:
    """Every number printed in the text as a float ('2,540' -> 2540, '1,6' -> 1.6, '3,060.00' -> 3060)."""
    out = set()
    for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        tok = tok.rstrip(",")
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", tok):
            out.add(float(tok.replace(",", "")))
        elif re.fullmatch(r"\d+,\d{1,2}", tok):
            out.add(float(tok.replace(",", ".")))
        else:
            out.update(float(t) for t in re.findall(r"\d+(?:\.\d+)?", tok))
    return out


def validate(v: Visit, text: str) -> tuple[Visit, list[str]]:
    """Null every extracted figure that does not appear in the report. Returns (visit, issues)."""
    issues = []
    present = number_values(text)

    def check(name: str, value):
        if value is None:
            return None
        if any(abs(value - x) < 1e-6 for x in present):
            return value
        issues.append(f"{name}={value:g} not found in report text; set to null")
        return None

    for f in ("onsite_hours", "travel_hours", "total_cost", "total_ex_vat", "labour_rate", "travel_rate", "quantity_replaced"):
        setattr(v, f, check(f, getattr(v, f)))
    for p in v.parts:
        p.unit_price = check(f"unit_price({p.name})", p.unit_price)
        p.line_total = check(f"line_total({p.name})", p.line_total)
    if v.cost_text and _squash(v.cost_text) not in _squash(text):
        issues.append("cost_text is not a verbatim quote; cleared")
        v.cost_text = None
    if v.total_cost is not None and v.currency is None:
        issues.append("total_cost has no currency")
    return v, issues


def _to_csv(rows: list[dict]) -> None:
    flat = [{**r, "job_tags": ";".join(r["job_tags"]), "parts": json.dumps(r["parts"]), "issues": "; ".join(r["issues"])}
            for r in rows]
    with open(CACHE_DIR / "service_visits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)


def extract_all() -> list[dict]:
    """Extract every report once. Progress is saved after each report, so a failure part-way
    (quota, network) loses nothing: run again and it resumes."""
    from src.grounded import _generate  # reuse the retrying Gemini helper

    CACHE_DIR.mkdir(exist_ok=True)
    done = {}
    if PARTIAL.exists():
        done = {r["report_id"]: r for r in map(json.loads, filter(str.strip, PARTIAL.read_text().splitlines()))}
    paths = sorted(REPORT_DIR.glob("FSR-*.txt"))
    if done:
        log.info("Resuming: %d of %d reports already extracted.", len(done), len(paths))
    rows = []
    with open(PARTIAL, "a") as partial:
        for path in paths:
            if path.stem in done:
                rows.append(done[path.stem])
                continue
            text = path.read_text(encoding="utf-8")
            try:
                visit = _generate(f"REPORT:\n{text}", SYSTEM, Visit)
            except llm.LLMUnavailable:
                log.error("Stopped after %d of %d reports (model unavailable). Progress is saved; run again to resume.",
                          len(rows), len(paths))
                raise
            visit, issues = validate(visit, text)
            row = {"report_id": path.stem, **visit.model_dump(), "issues": issues}
            rows.append(row)
            partial.write(json.dumps(row) + "\n")
            partial.flush()
            log.info("%s %s %s total=%s %s%s", path.stem, visit.equipment, visit.country or "?", visit.total_cost,
                     visit.currency or "", f"  !! {issues}" if issues else "")
    TABLE.write_text(json.dumps(rows, indent=1))
    _to_csv(rows)
    PARTIAL.unlink()
    return rows


def build_if_missing() -> bool:
    """First-run helper: build the table if it does not exist. Returns False if it could not be built."""
    if TABLE.exists():
        return True
    log.info("Building the service-visits table once (%d reports, one model call each)...",
             len(list(REPORT_DIR.glob("FSR-*.txt"))))
    try:
        extract_all()
        return True
    except llm.LLMUnavailable:
        return False


def load_visits() -> list[dict]:
    if not TABLE.exists():
        raise FileNotFoundError("Run `python -m src.ingest.services` first to build the visits table.")
    return json.loads(TABLE.read_text())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = extract_all()
    log.info("%d reports; %d state a total cost; %d had values nulled by validation", len(rows),
             sum(r["total_cost"] is not None for r in rows), sum(bool(r["issues"]) for r in rows))
