"""Odds math and the edge finder."""
from __future__ import annotations

import pandas as pd

from . import config
from .model import OUT_STATUSES, Model, check_period, prob_over_under


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def implied_prob(odds: float) -> float:
    return 1 / american_to_decimal(odds)


def no_vig(over_odds, under_odds) -> tuple[float, float]:
    """Strip the bookmaker margin proportionally."""
    po, pu = implied_prob(over_odds), implied_prob(under_odds)
    return po / (po + pu), pu / (po + pu)


def expected_value(p_win: float, p_push: float, odds: float) -> float:
    """EV per $1 staked."""
    return p_win * (american_to_decimal(odds) - 1) - (1 - p_win - p_push)


def kelly_stake(p_win: float, p_push: float, odds: float, bankroll: float) -> float:
    b = american_to_decimal(odds) - 1
    p_loss = 1 - p_win - p_push
    f = (b * p_win - p_loss) / b if b > 0 else 0
    f = max(0.0, f) * config.KELLY_FRACTION
    return round(min(f, config.MAX_STAKE_PCT) * bankroll, 2)


def find_edges(model: Model, props: pd.DataFrame, bankroll: float, ctx=None) -> pd.DataFrame:
    """props columns: player, opp, stat, line, over_odds, [under_odds], [period], [minutes].
    period is q1 (default) or full. minutes overrides projected minutes for that period
    (the old column name q1_min still works)."""
    rows = []
    props = props.copy()
    if "minutes" not in props.columns:
        props["minutes"] = props["q1_min"] if "q1_min" in props.columns else None
    if "period" not in props.columns:
        props["period"] = "q1"
    for _, r in props.iterrows():
        try:
            period = check_period(r["period"] if pd.notna(r["period"]) else "q1")
            pr = model.project(r["player"], r["opp"], r["stat"], period, r["minutes"], ctx=ctx)
        except (KeyError, ValueError) as e:
            rows.append({"player": r["player"], "stat": r["stat"], "period": r.get("period"),
                         "note": str(e).strip("'\"")})
            continue
        p_over, p_under, p_push = prob_over_under(pr["mean"], pr["alpha"], float(r["line"]))
        has_under = pd.notna(r.get("under_odds"))
        fair_over, fair_under = (no_vig(r["over_odds"], r["under_odds"]) if has_under
                                 else (implied_prob(r["over_odds"]), None))
        for side, p, odds, fair in [("over", p_over, r["over_odds"], fair_over),
                                    ("under", p_under, r.get("under_odds"), fair_under)]:
            if odds is None or pd.isna(odds):
                continue
            ev = expected_value(p, p_push, odds)
            edge = p - fair if fair is not None else None
            notes = []
            if pr["games"] < 10:
                notes.append(f"only {pr['games']} games")
            if pr["start_rate_last10"] < 0.8:
                notes.append("not a regular starter")
            trend = 1.5 if period == "q1" else 4.0
            if abs(pr["min_last5"] - pr["min_season"]) > trend:
                notes.append(f"minutes trending {pr['min_last5']:.1f} (last 5)")
            if pr["status"]:
                notes.insert(0, f"listed {pr['status']}")
            if pr["b2b"]:
                notes.append("back-to-back")
            if pr["opp_b2b"]:
                notes.append("opp on back-to-back")
            if pr["teammates_out"]:
                notes.append("teammates out: " + ", ".join(pr["teammates_out"][:3]))
            if edge is not None and edge > 0.12:
                notes.append("edge this big usually means the book knows something "
                             "(injury, minutes limit, role change) - verify first")
            bet = pr["status"] not in OUT_STATUSES and ev >= config.MIN_EV and (edge is None or edge >= config.MIN_EDGE)
            rows.append({
                "period": period, "player": pr["player_name"], "team": pr["team"],
                "pos": pr["pos"], "opp": pr["opp"], "stat": pr["stat"], "line": r["line"],
                "side": side, "odds": int(odds), "proj": round(pr["mean"], 2),
                "minutes": round(pr["minutes"], 1),
                "season_avg": round(pr["season_avg"], 2),
                "last10": round(pr["last10_avg"], 2),
                "vs_opp": (f"{pr['vs_avg']:.1f} in {pr['vs_games']}g" if pr["vs_games"] else "-"),
                "matchup": round(pr["matchup_factor"], 2),
                "h2h": round(pr["h2h_factor"], 2),
                "model_p": round(p, 3),
                "fair_p": round(fair, 3) if fair is not None else None,
                "edge": round(edge, 3) if edge is not None else None,
                "ev": round(ev, 3),
                "stake": kelly_stake(p, p_push, odds, bankroll) if bet else 0.0,
                "bet": "YES" if bet else "",
                "note": "; ".join(notes),
            })
    out = pd.DataFrame(rows)
    if "ev" in out:
        out = out.sort_values(["bet", "ev"], ascending=[False, False], na_position="last")
    return out
