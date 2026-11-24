"""Walk-forward backtest: for every game day, fit only on earlier days, project
that day's starters, and compare with what actually happened.

There are no historical prop lines in the free data, so this measures projection
accuracy and probability calibration, not betting profit. The 'proxy line' is the
player's season average rounded to a half point, roughly where books hang lines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .model import (Model, PERIOD_LABELS, add_combos, check_period, effective_alpha, mark_b2b,
                    prob_over_under, team_game_dates)

FULL = "full model"


def _variants(period: str, has_full: bool) -> dict:
    off = dict(use_recent=False, use_b2b=False)
    v = {"season avg only": None}
    if period == "q1":
        v["Q1 min x Q1 rate"] = dict(use_full=False, use_matchup=False, use_h2h=False, **off)
        if has_full:
            v["+ full-game rates"] = dict(use_full=True, use_matchup=False, use_h2h=False, **off)
        v["+ matchup (by position)"] = dict(use_full=has_full, use_matchup=True, use_h2h=False, **off)
    else:
        v["min x rate"] = dict(use_matchup=False, use_h2h=False, **off)
        v["+ matchup (by position)"] = dict(use_matchup=True, use_h2h=False, **off)
    v["+ head-to-head"] = dict(use_full=True, use_matchup=True, use_h2h=True, **off)
    v["+ last 10 games"] = dict(use_full=True, use_matchup=True, use_h2h=True,
                                use_recent=True, use_b2b=False)
    v[FULL + " (+ back-to-back)"] = dict(use_full=True, use_matchup=True, use_h2h=True, use_recent=True, use_b2b=True)
    return v


def _collect(q1, full, period, season, dates, stats, variants):
    tbl = add_combos((q1 if period == "q1" else full))
    tbl = mark_b2b(tbl[tbl["season"] == season], team_game_dates(q1, full))
    frames = []
    for d in dates:
        hq = q1[q1["game_date"] < d]
        hf = full[full["game_date"] < d] if full is not None else None
        src = hq if period == "q1" else hf
        if src[src["season"] == season].empty:
            continue
        day = tbl[(tbl["game_date"] == d) & tbl["starter"]]
        if day.empty:
            continue
        res = day[["player_id", "opp", *stats]].rename(columns={s: f"act_{s}" for s in stats})
        base = None
        for name, kw in variants.items():
            if kw is None:
                continue
            m = Model(hq, hf, season, **kw)
            p = m.project_many(day[["player_id", "opp", "b2b", "opp_b2b"]], period)
            cols = {f"proj_{s}": f"{name}|{s}" for s in stats}
            p = p[["player_id", "n_eff", "min_avg", *[f"avg_{c}" for c in config.BASE_STATS], *cols]]
            p = p.rename(columns=cols)
            if base is None:
                base = p[["player_id", "n_eff"]].copy()
                for s in stats:
                    parts = config.COMBOS.get(s, [s])
                    base[f"season avg only|{s}"] = sum(p[f"avg_{c}"] for c in parts)
                alphas = m.alpha[period]
            base = base.merge(p[["player_id", *cols.values()]], on="player_id")
        if base is not None and len(base):
            frames.append(res.merge(base, on="player_id"))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), alphas if frames else {}


def run(q1: pd.DataFrame, full: pd.DataFrame | None = None, period: str = "q1",
        season: str | None = None, start: str | None = None, stats: list[str] | None = None,
        every: int = 1) -> str:
    period = check_period(period)
    if period == "full" and full is None:
        return "No full-game stats downloaded yet - update stats first."
    season = season or q1["season"].max()
    stats = stats or ["pts", "reb", "ast", "fg3m", "pra"]
    src = q1 if period == "q1" else full
    dates = sorted(src.loc[src["season"] == season, "game_date"].unique())
    dates = [d for d in dates if d >= pd.Timestamp(start)] if start else dates[14:]
    dates = dates[::every]
    if not dates:
        return "No dates to backtest yet (needs a few weeks of games)."
    variants = _variants(period, full is not None)
    df, alphas = _collect(q1, full, period, season, dates, stats, variants)
    if df.empty:
        return "Not enough games to backtest yet."

    out = [f"{PERIOD_LABELS[period]} backtest, {season}: {len(dates)} game days, {len(df)} "
           f"starter-games (each projected using only earlier games)\n",
           "Mean absolute error (lower is better; each row adds one input):"]
    mae = pd.DataFrame({s: {v: (df[f"{v}|{s}"] - df[f"act_{s}"]).abs().mean() for v in variants}
                        for s in stats})
    out.append(mae.round(3).to_string())

    full_name = FULL + " (+ back-to-back)"
    out.append(f"\nCalibration of the full model, P(over) at proxy line "
               f"(season avg rounded to .5):")
    cal = []
    for s in stats:
        line = np.floor(df[f"season avg only|{s}"]) + 0.5
        p = np.array([prob_over_under(mu, effective_alpha(alphas[s], mu, n), L)[0]
                      for mu, L, n in zip(df[f"{full_name}|{s}"], line, df["n_eff"])])
        hit = (df[f"act_{s}"] > line).astype(float).values
        bins = pd.cut(p, [0, .35, .45, .55, .65, 1], include_lowest=True)
        for b, grp in pd.DataFrame({"p": p, "hit": hit, "bin": bins}).groupby("bin", observed=True):
            cal.append({"stat": s, "model P(over)": str(b), "n": len(grp),
                        "avg predicted": round(grp["p"].mean(), 3),
                        "actual over rate": round(grp["hit"].mean(), 3)})
        cal.append({"stat": s, "model P(over)": "Brier score", "n": len(p),
                    "avg predicted": round(float(np.mean((p - hit) ** 2)), 4), "actual over rate": ""})
    out.append(pd.DataFrame(cal).to_string(index=False))
    out.append("\nWhat to look for: 'avg predicted' close to 'actual over rate' means the "
               "probabilities can be trusted. If an input doesn't lower the error, turn its "
               "weight down in config.py (or run Tune).")
    return "\n".join(out)


def tune(q1: pd.DataFrame, full: pd.DataFrame | None = None, period: str = "q1",
         season: str | None = None, every: int = 3) -> str:
    """Grid-search the main weights for one period on your own data."""
    period = check_period(period)
    if period == "full" and full is None:
        return "No full-game stats downloaded yet."
    season = season or q1["season"].max()
    src = q1 if period == "q1" else full
    dates = sorted(src.loc[src["season"] == season, "game_date"].unique())[14::every]
    if len(dates) < 5:
        return "Not enough game days yet to tune (need a few weeks of games)."
    stats = ["pts", "reb", "ast", "pra"]
    tbl = add_combos(src[src["season"] == season])
    off = 1e9

    if period == "q1" and full is not None:
        grid = [("ANCHOR_MINUTES_Q1", None, [20, 60, 150, 400]),
                ("OPP_SHRINK_GAMES", "q1", [10, 30, 80, off])]
    elif period == "q1":
        grid = [("H2H_SHRINK_GAMES", "q1", [5, 20, 60, off]),
                ("OPP_SHRINK_GAMES", "q1", [10, 30, 80, off])]
    else:
        grid = [("H2H_SHRINK_GAMES", "full", [3, 10, 30, off]),
                ("OPP_SHRINK_GAMES", "full", [5, 20, 60, off])]

    def get(name, key):
        v = getattr(config, name)
        return v[key] if key else v

    def put(name, key, val):
        if key:
            getattr(config, name)[key] = val
        else:
            setattr(config, name, val)

    saved = [get(n, k) for n, k, _ in grid]
    rows = []
    try:
        for a in grid[0][2]:
            for b in grid[1][2]:
                put(grid[0][0], grid[0][1], a)
                put(grid[1][0], grid[1][1], b)
                errs = {s: [] for s in stats}
                for d in dates:
                    hq = q1[q1["game_date"] < d]
                    hf = full[full["game_date"] < d] if full is not None else None
                    m = Model(hq, hf, season)
                    day = tbl[(tbl["game_date"] == d) & tbl["starter"]]
                    p = m.project_many(day[["player_id", "opp"]], period).merge(
                        day[["player_id", *stats]].rename(columns={s: f"a_{s}" for s in stats}),
                        on="player_id")
                    for s in stats:
                        errs[s].append((p[f"proj_{s}"] - p[f"a_{s}"]).abs())
                off_label = ["off", "full-game only" if (period == "q1" and full is not None) else "off"]
                row = {grid[0][0]: off_label[0] if a >= off else a,
                       grid[1][0]: off_label[1] if b >= off else b}
                row.update({f"MAE {s}": round(pd.concat(v).mean(), 4) for s, v in errs.items()})
                rows.append(row)
    finally:
        for (n, k, _), v in zip(grid, saved):
            put(n, k, v)
    df = pd.DataFrame(rows)
    df["avg rank"] = df[[c for c in df if c.startswith("MAE")]].rank().mean(axis=1)
    df = df.sort_values("avg rank")
    best = df.iloc[0]
    keys = [f"{n}" + (f'["{k}"]' if k else "") for n, k, _ in grid]
    return (f"{PERIOD_LABELS[period]} tuning:\n" + df.to_string(index=False) +
            f"\n\nBest: {keys[0]} = {best[grid[0][0]]}, {keys[1]} = {best[grid[1][0]]}  "
            "('off' = input not used; 'full-game only' = use the full-game matchup rating as is). "
            "Put these in q1model/config.py (use 1000000000 for off). "
            "Differences under ~0.01 are noise.")
