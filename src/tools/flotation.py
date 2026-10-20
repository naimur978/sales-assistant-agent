"""Assess a customer's concentrate readings against the plant's hourly history. Everything is computed
from the CSV with pandas; nothing comes from general knowledge.

assess(silica, iron) answers three things, all by counting and comparing, with no prediction model:
  1. where do these readings sit in the history (percentiles, how often, what is typical at this iron grade)
  2. how do hours this poor come in time (how many episodes, how long they last)
  3. what do the best hours look like, and does that picture hold in every period and feed group?
"""
from functools import lru_cache

import pandas as pd
from pydantic import BaseModel, Field

from src.config import FLOTATION_CSV

SI, FE = "% Silica Concentrate", "% Iron Concentrate"
MIN_GROUP = 30  # hours needed in a group before comparing it
MIN_EFFECT = 0.3  # standardised difference a variable needs, in every group, to count as consistent
LONG_EPISODE_H = 6


class Readings(BaseModel):
    silica: float = Field(ge=0, le=100, description="% silica in the concentrate")
    iron: float = Field(ge=0, le=100, description="% iron in the concentrate")
    silica_max: float | None = Field(default=None, ge=0, le=100, description="the customer's silica limit, %")
    iron_min: float | None = Field(default=None, ge=0, le=100, description="the customer's iron minimum, %")


@lru_cache
def load() -> pd.DataFrame:
    return pd.read_csv(FLOTATION_CSV, parse_dates=["date"])


def _band(df: pd.DataFrame, col: str, centre: float, half: float) -> pd.Series:
    return df.loc[(df[col] - centre).abs() <= half]


def position(r: Readings, df: pd.DataFrame) -> dict:
    s, f = df[SI], df[FE]
    near_fe, near_si = _band(df, FE, r.iron, 0.25), _band(df, SI, r.silica, 0.15)
    recent = df[df.date >= df.date.max() - pd.Timedelta(days=30)]
    inside = f.min() <= r.iron <= f.max() and s.min() <= r.silica <= s.max()
    q = lambda x: {"p10": x.quantile(.1), "median": x.median(), "p90": x.quantile(.9)}
    spec = None
    if r.silica_max is not None or r.iron_min is not None:
        ok_s = (s <= r.silica_max) if r.silica_max is not None else pd.Series(True, index=df.index)
        ok_f = (f >= r.iron_min) if r.iron_min is not None else pd.Series(True, index=df.index)
        spec = {"silica_ok": None if r.silica_max is None else bool(r.silica <= r.silica_max),
                "iron_ok": None if r.iron_min is None else bool(r.iron >= r.iron_min),
                "plant_hours_in_spec": float((ok_s & ok_f).mean())}
    return {
        "spec": spec, "n_hours": len(df), "from": df.date.min(), "to": df.date.max(),
        "silica_share_at_or_above": float((s >= r.silica).mean()),
        "iron_share_at_or_below": float((f <= r.iron).mean()),
        "both_share": float(((s >= r.silica) & (f <= r.iron)).mean()),
        "silica_typical": q(s), "iron_typical": q(f),
        "silica_given_iron": {**q(near_fe[SI]), "n": len(near_fe), "share_at_or_above": float((near_fe[SI] >= r.silica).mean())} if len(near_fe) else None,
        "iron_given_silica": {**q(near_si[FE]), "n": len(near_si)} if len(near_si) else None,
        "recent_30d": {"silica_mean": recent[SI].mean(), "iron_mean": recent[FE].mean(), "n": len(recent)},
        "inside_observed_range": bool(inside),
        "iron_silica_correlation": float(f.corr(s)),
    }


def episodes(silica: float) -> dict:
    """Runs of consecutive hours at or above the customer's silica reading, never joined across the data gap."""
    df = load()
    gap = df.date.diff().dt.total_seconds().div(3600).fillna(1)
    segment = (gap > 1).cumsum()
    high = df[SI] >= silica
    run = (high != high.shift()).cumsum() + segment * 100000
    lengths = df[high].groupby(run[high]).size()
    if lengths.empty:
        return {"count": 0}
    return {"count": int(len(lengths)), "median_h": float(lengths.median()), "mean_h": float(lengths.mean()),
            "longest_h": int(lengths.max()), "share_long": float((lengths >= LONG_EPISODE_H).mean()),
            "share_hours": float(high.mean())}


def _effects(df: pd.DataFrame, silica: float, sd: pd.Series, controls: list[str]):
    """Standardised difference (best hours minus poor hours) for each variable, or None if a side is too small."""
    best = df[(df[SI] <= load()[SI].quantile(.25)) & (df[FE] >= load()[FE].median())]
    poor = df[df[SI] >= silica]
    if len(best) < MIN_GROUP or len(poor) < MIN_GROUP:
        return None
    return (best[controls].mean() - poor[controls].mean()) / sd


def good_hours(silica: float) -> dict:
    """What the best hours (low silica, high iron) look like against hours as poor as the customer's, and whether
    each difference holds in all three time periods and all three feed-silica groups."""
    df = load()
    controls = [c for c in df.columns if c not in ("date", SI, FE, "% Iron Feed", "% Silica Feed")]
    sd = df[controls].std()
    best = df[(df[SI] <= df[SI].quantile(.25)) & (df[FE] >= df[FE].median())]
    poor = df[df[SI] >= silica]
    if silica <= df[SI].median() or len(best) < MIN_GROUP or len(poor) < MIN_GROUP:
        return {"meaningful": False, "silica": silica, "n_best": len(best), "n_poor": len(poor)}
    overall = (best[controls].mean() - poor[controls].mean()) / sd
    d = df.assign(period=pd.qcut(df.date.rank(method="first"), 3, labels=False),
                  feed=pd.qcut(df["% Silica Feed"], 3, labels=False))
    groups = [d[d[k] == v] for k in ("period", "feed") for v in range(3)]
    effects = [e for e in (_effects(g, silica, sd, controls) for g in groups) if e is not None]
    stable = [c for c in controls if len(effects) == 6 and all(abs(e[c]) >= MIN_EFFECT for e in effects)
              and len({1 if e[c] > 0 else -1 for e in effects}) == 1]
    top = overall.abs().sort_values(ascending=False).head(4).index
    return {"meaningful": True, "silica": silica, "n_best": len(best), "n_poor": len(poor), "groups_compared": len(effects),
            "rows": [(c, float(best[c].median()), float(poor[c].median()), float(overall[c])) for c in top],
            "consistent": stable}


def assess(r: Readings) -> dict:
    return {"readings": r.model_dump(), "position": position(r, load()), "episodes": episodes(r.silica),
            "good_hours": good_hours(r.silica),
            "caveats": ["One plant, hourly averages, March-September 2017 (one 13-day gap); the customer's plant may differ.",
                        "These are patterns in operating data, not proof of cause: operators adjust settings in response to quality.",
                        "'Normal' here means 'common in this plant's history', not 'meets the customer's specification'."]}


def _spec_lines(r: dict, p: dict, pct) -> list[str]:
    """When the customer has named their own limits, the answer is measured against those, not against the plant history."""
    sp = p["spec"]
    clauses, misses = [], []
    if sp["silica_ok"] is not None:
        clauses.append(f"silica {r['silica']:g}% is {'within' if sp['silica_ok'] else 'above'} their {r['silica_max']:g}% limit")
        misses += [] if sp["silica_ok"] else ["silica"]
    if sp["iron_ok"] is not None:
        clauses.append(f"iron {r['iron']:g}% is {'at or above' if sp['iron_ok'] else 'below'} their {r['iron_min']:g}% minimum")
        misses += [] if sp["iron_ok"] else ["iron"]
    head = "Out of their target." if misses else "Within their target."
    return [f"- {head} {clauses[0][0].upper() + clauses[0][1:]}" + (f", and {clauses[1]}." if len(clauses) > 1 else "."),
            f"- In the plant data, {pct(sp['plant_hours_in_spec'])} of hours meet that target."]


def _quick_answer(r: dict, p: dict, g: dict, pct) -> list[str]:
    """The lines a salesperson can read on the phone. Everything after 'Details' backs them up."""
    ts, tf = p["silica_typical"]["median"], p["iron_typical"]["median"]
    if not g["meaningful"]:
        cannot = "- Their silica is too typical to compare against better hours, so this data gives no setting advice."
    elif g["consistent"]:
        cannot = f"- {', '.join(g['consistent'])} separate good hours from poor ones in every period, so check them first."
    else:
        cannot = "- No operating variable separates good hours from poor ones in every period, so this data cannot tell us which setting to change."
    if p.get("spec"):
        return ["Quick answer:", *_spec_lines(r, p, pct), cannot]
    worse_si, worse_fe = r["silica"] > ts, r["iron"] < tf
    verdict = (f"Not unusual, but not good either. About {pct(p['both_share'])} of hours are this poor or worse." if worse_si and worse_fe
               else "Mixed. One reading is worse than typical and the other is not." if worse_si or worse_fe
               else "Yes, that is normal. Their readings are typical or better.")
    return ["Quick answer:",
            f"- {verdict}",
            f"- Typical is {ts:.1f}% silica and {tf:.1f}% iron, against their {r['silica']:g}% and {r['iron']:g}%.",
            cannot]


def render(a: dict) -> str:
    r, p, e, g = a["readings"], a["position"], a["episodes"], a["good_hours"]
    pct = lambda x: f"{100 * x:.0f}%"
    quick = _quick_answer(r, p, g, pct)
    asks = (["When did it start, and what changed around then?"] if p.get("spec")
            else ["What is their silica target?", "Does it keep happening, or was it just today?"])
    out = [*quick, "", "Worth asking:", *[f"- {q}" for q in asks], "", "Details:", "",
           f"Where it sits (against {p['n_hours']:,} hourly readings, {p['from']:%b %Y} to {p['to']:%b %Y}):",
           f"- Silica {r['silica']:g}% or worse in {pct(p['silica_share_at_or_above'])} of hours "
           f"(middle 80% of hours {p['silica_typical']['p10']:.1f}-{p['silica_typical']['p90']:.1f}%).",
           f"- Iron {r['iron']:g}% or lower in {pct(p['iron_share_at_or_below'])} of hours "
           f"(middle 80% {p['iron_typical']['p10']:.1f}-{p['iron_typical']['p90']:.1f}%)."]
    if not p["inside_observed_range"]:
        out.append("- WARNING: these readings are outside the range seen in the data; treat the comparison with caution.")
    sg = p["silica_given_iron"]
    if sg:
        out.append(f"- At iron near {r['iron']:g}%, silica is usually {sg['median']:.1f}% ({sg['p10']:.1f}-{sg['p90']:.1f}); "
                   f"{r['silica']:g}% or higher happens in {pct(sg['share_at_or_above'])} of those hours.")
    rc = p["recent_30d"]
    out.append(f"- Last 30 days of data, mean silica {rc['silica_mean']:.2f}%, mean iron {rc['iron_mean']:.2f}%.")
    if e["count"]:
        out += ["", f"How hours at {r['silica']:g}% silica or more come in time:",
                f"- {e['count']} separate episodes. A typical one lasts {e['median_h']:.0f} h, the average is {e['mean_h']:.1f} h, "
                f"the longest {e['longest_h']} h, and {pct(e['share_long'])} last {LONG_EPISODE_H} h or more."]
    out += ["", "What the best hours look like (low silica, high iron):"]
    if g["meaningful"]:
        out.append(f"- Compared with {g['n_poor']:,} hours at {g['silica']:g}% silica or more (best hours versus poor hours, median):")
        out += [f"    {name}: {b:,.1f} vs {pr:,.1f}" for name, b, pr, _ in g["rows"][:3]]
        out.append(f"- Held in every period and feed group: {', '.join(g['consistent']) if g['consistent'] else 'nothing, so the pattern changes over time'}.")
    else:
        out.append(f"- Their silica of {g['silica']:g}% is too typical in the data to compare against better hours.")
    out += ["", "Keep in mind:"] + [f"- {c}" for c in a["caveats"]]
    return "\n".join(out)
