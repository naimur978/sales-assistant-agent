"""Rebuild spec tables from the brochure PDFs, because the provided .md text has them flattened.

Each table becomes self-contained lines (series / model + column header + value + unit on one line), appended to
its page chunk, so a number is never separated from the label it belongs to.
"""
import json
import re
from pathlib import Path

CACHE = Path(".cache/tables.json")
HEADER = re.compile(r"(?:larox )?pf [\d.]+ (?:series|model)|frame size|type/area", re.I)
JUNK = re.compile(r"\(cid:\d+\)")


def _cell(v) -> str:
    return re.sub(r"\s+", " ", v or "").strip()


def _clean_rows(rows: list[list]) -> list[list[str]]:
    rows = [[_cell(c) for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows or any(JUNK.search(c) for r in rows for c in r):
        return []  # font-encoding garbage or an empty drawing, not a table
    return rows


def _series_lines(rows: list[list[str]]) -> list[str]:
    """Larox style: header row = series name + sizes, then 'label | unit | one value per size'."""
    head = next(i for i, r in enumerate(rows) if HEADER.match(r[0]))
    series = rows[head][0]
    if series.lower().startswith("frame size"):  # datasheet: the sizes are the 'Chambers' row below
        head += 1
        series = f"Frame size table ({rows[head][0]})"
    elif series.lower().startswith("type/area"):
        series = "Type/area table"
    sizes = rows[head][2:]
    lines = []
    for r in rows[head + 1:]:
        label, unit, vals = r[0], r[1], r[2:]
        if not label or not any(vals):
            continue
        for size, v in zip(sizes, vals):
            if size and v:
                lines.append(f"{series}, size {size}: {label} = {v} {unit}".strip())
        if len([v for v in vals if v]) == 1:  # a single value spans every size (e.g. plate size)
            lines.append(f"{series}: {label} = {next(v for v in vals if v)} {unit}".strip())
    return lines


def _model_lines(rows: list[list[str]]) -> list[str]:
    """Pump style: 'Model' header over up to 3 header rows, then one row per model."""
    start = next(i for i, r in enumerate(rows) if r[0].lower().startswith("model"))
    body = start + 1
    while body < len(rows) and not re.search(r"\d", rows[body][0]):
        body += 1
    head_rows = rows[start:body]
    width = max(len(r) for r in head_rows)
    heads = []
    for j in range(1, width):
        parts, last = [], None
        for r in head_rows:
            v = r[j] if j < len(r) else ""
            if not v:  # a spanning header applies to the columns to its right
                v = next((r[k] for k in range(j, 0, -1) if k < len(r) and r[k]), "") if r is not head_rows[-1] else ""
            if v and v != last:
                parts.append(v)
                last = v
        heads.append(" ".join(parts))
    lines = []
    for r in rows[body:]:
        if not r[0] or len(r[0]) > 20:  # footnotes, not a model row
            continue
        cells = [f"{h} = {v}" for h, v in zip(heads, r[1:]) if v and h]
        lines.append(f"{r[0]}: " + "; ".join(cells))
    return lines


def table_lines(rows: list[list]) -> list[str]:
    rows = _clean_rows(rows)
    if len(rows) < 3 or max(len(r) for r in rows) < 4:
        return []
    if any(r[0].lower().startswith("model") for r in rows[:3]):
        return _model_lines(rows)
    if any(HEADER.match(r[0]) for r in rows[:3]):
        return _series_lines(rows)
    return []  # unknown layout: leave the page text alone rather than guess


def extract(brochure_dir: Path) -> dict[str, list[str]]:
    """{'file:page': [lines]} for every page with a table that parsed cleanly."""
    import pdfplumber

    out: dict[str, list[str]] = {}
    for path in sorted(Path(brochure_dir).glob("*.pdf")):
        with pdfplumber.open(path) as pdf:
            for no, page in enumerate(pdf.pages, 1):
                lines = []
                for t in page.find_tables():
                    try:
                        lines += table_lines(t.extract())
                    except (StopIteration, IndexError):
                        continue
                if lines:
                    out[f"{path.stem}:{no}"] = lines
    return out


def load_tables(brochure_dir: Path) -> dict[str, list[str]]:
    """Cached after the first run of the PDFs; delete .cache/tables.json to rebuild."""
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        pass
    try:
        tables = extract(brochure_dir)
    except ImportError:
        return {}  # pdfplumber missing: fall back to the flattened text
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(tables, indent=1))
    return tables
