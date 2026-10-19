"""Assess a customer's concentrate readings against the plant's hourly history. Everything is computed
from the CSV with pandas; nothing comes from general knowledge.

assess(silica, iron) answers three things:
  1. where do these readings sit in the history (percentiles, how often, what is typical at this iron grade)
  2. what differs in hours with high silica (associations in the process variables)
  3. how far can those associations be trusted (does a model built on early data predict later data?)
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from src.config import FLOTATION_CSV

SI, FE = "% Silica Concentrate", "% Iron Concentrate"
TRUSTED_R2 = 0.30  # below this, process variables do not predict silica well enough to recommend changes


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


@lru_cache
def drivers() -> dict:
    df = load()
    X, y = df.drop(columns=["date", SI, FE]), df[SI]
    n, cut = len(df), int(len(df) * 0.7)

    def ols(Xa, ya):
        mu, sd = Xa.mean(), Xa.std()
        Z = np.c_[np.ones(len(Xa)), ((Xa - mu) / sd).values]
        return np.linalg.lstsq(Z, ya.values, rcond=None)[0], mu, sd

    b, mu, sd = ols(X[:cut], y[:cut])
    ols_r2 = r2_score(y[cut:], np.c_[np.ones(n - cut), ((X[cut:] - mu) / sd).values] @ b)
    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0).fit(X[:cut], y[:cut])
    gbm_r2 = r2_score(y[cut:], gbm.predict(X[cut:]))

    coefs = pd.DataFrame([ols(X.iloc[i], y.iloc[i])[0][1:] for i in np.array_split(np.arange(n), 5)], columns=X.columns)
    full = pd.Series(ols(X, y)[0][1:], index=X.columns)
    stable = [(c, float(full[c])) for c in X.columns
              if (np.sign(coefs[c]) == np.sign(full[c])).all()]
    stable.sort(key=lambda t: -abs(t[1]))

    hi, lo = df[df[SI] >= 3.1], df[df[SI] < 2.0]
    diff = ((hi[X.columns].mean() - lo[X.columns].mean()) / X.std()).sort_values(key=abs, ascending=False)
    return {
        "test_r2_linear": float(ols_r2), "test_r2_gbm": float(gbm_r2), "train_hours": cut, "test_hours": n - cut,
        "consistent_in_all_5_blocks": stable,
        "high_vs_low_silica": [(c, float(v), float(hi[c].mean()), float(lo[c].mean())) for c, v in diff.head(6).items()],
        "n_high": len(hi), "n_low": len(lo),
        "trusted": max(ols_r2, gbm_r2) >= TRUSTED_R2,
    }


def assess(r: Readings) -> dict:
    return {"readings": r.model_dump(), "position": position(r, load()), "drivers": drivers(),
            "caveats": ["One plant, hourly averages, March-September 2017 (one 13-day gap); the customer's plant may differ.",
                        "Associations in operating data are not proof of cause: operators adjust settings in response to quality.",
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


def _quick_answer(r: dict, p: dict, d: dict, pct) -> list[str]:
    """The lines a salesperson can read on the phone. Everything after 'Details' backs them up."""
    ts, tf = p["silica_typical"]["median"], p["iron_typical"]["median"]
    cannot = ("- This data cannot tell us which setting to change." if not d["trusted"]
              else "- Some process settings are linked to silica, see the details.")
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
    r, p, d = a["readings"], a["position"], a["drivers"]
    pct = lambda x: f"{100 * x:.0f}%"
    quick = _quick_answer(r, p, d, pct)
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
    g = p["silica_given_iron"]
    if g:
        out.append(f"- At iron near {r['iron']:g}%, silica is usually {g['median']:.1f}% ({g['p10']:.1f}-{g['p90']:.1f}); "
                   f"{r['silica']:g}% or higher happens in {pct(g['share_at_or_above'])} of those hours.")
    rc = p["recent_30d"]
    out.append(f"- Last 30 days of data, mean silica {rc['silica_mean']:.2f}%, mean iron {rc['iron_mean']:.2f}%.")
    out += ["", "What the process data says about causes:",
            f"- Process settings predict silica {'reasonably well' if d['trusted'] else 'no better than guessing the average'} "
            f"(R2 {d['test_r2_linear']:.2f} linear, {d['test_r2_gbm']:.2f} gradient boosting, tested on later hours)."
            + ("" if d["trusted"] else " So there is no firm advice on which setting to change."),
            f"- Hours with silica of 3.1% or more ({d['n_high']:,}) versus below 2.0% ({d['n_low']:,}) differ most in (high-silica hours vs low):"]
    for c, sd_diff, hi, lo in d["high_vs_low_silica"][:2]:
        out.append(f"    {c}, {hi:,.1f} vs {lo:,.1f}")
    out += ["", "Keep in mind:"] + [f"- {c}" for c in a["caveats"]]
    return "\n".join(out)
