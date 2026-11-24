"""Bet log: record bets, settle them from Q1 box scores, report ROI and CLV."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .betting import american_to_decimal, implied_prob
from .model import add_combos, check_period, components, norm_name

COLS = ["bet_id", "placed_at", "game_date", "period", "player", "opp", "stat", "line", "side",
        "odds", "stake", "model_p", "proj", "close_odds", "actual", "result", "profit"]


def load(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path).reindex(columns=COLS)
        df["period"] = df["period"].fillna("q1")   # bets logged before full games existed
        return df
    return pd.DataFrame(columns=COLS)


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(path, index=False)


def add_bet(path: Path, **kw) -> int:
    df = load(path)
    bet_id = int(df["bet_id"].max()) + 1 if len(df) else 1
    components(kw["stat"])  # validates stat name
    kw["period"] = check_period(kw.get("period", "q1"))
    row = {c: None for c in COLS}
    row.update(kw, bet_id=bet_id, placed_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
               result="pending", stat=kw["stat"].lower(), side=kw["side"].lower())
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save(df, path)
    return bet_id


def set_close(path: Path, bet_id: int, close_odds: float):
    df = load(path)
    if bet_id not in set(df["bet_id"]):
        raise SystemExit(f"No bet {bet_id}")
    df.loc[df["bet_id"] == bet_id, "close_odds"] = close_odds
    save(df, path)


def settle(path: Path, q1_table: pd.DataFrame, full_table: pd.DataFrame | None = None) -> int:
    df = load(path)
    tables = {}
    for period, table in [("q1", q1_table), ("full", full_table)]:
        if table is None:
            continue
        t = add_combos(table)
        t["name_key"] = t["player_name"].map(norm_name)
        t["d"] = t["game_date"].dt.strftime("%Y-%m-%d")
        tables[period] = t
    n = 0
    for i, b in df[df["result"] == "pending"].iterrows():
        t = tables.get(b["period"])
        if t is None:
            continue
        hit = t[(t["name_key"] == norm_name(b["player"])) & (t["d"] == str(b["game_date"]))]
        if hit.empty:
            continue
        actual = float(hit.iloc[0][b["stat"]])
        line = float(b["line"])
        if actual == line:
            res, profit = "P", 0.0
        else:
            won = (actual > line) == (b["side"] == "over")
            res = "W" if won else "L"
            profit = float(b["stake"]) * (american_to_decimal(b["odds"]) - 1) if won else -float(b["stake"])
        df.loc[i, ["actual", "result", "profit"]] = [actual, res, round(profit, 2)]
        n += 1
    save(df, path)
    return n


def report(path: Path) -> str:
    df = load(path)
    if df.empty:
        return "No bets logged yet."
    done = df[df["result"].isin(["W", "L", "P"])].copy()
    lines = [f"Bets logged: {len(df)}  (settled {len(done)}, pending {len(df) - len(done)})"]
    if done.empty:
        return "\n".join(lines)
    staked = done["stake"].astype(float).sum()
    profit = done["profit"].astype(float).sum()
    w, l, p = (done["result"] == "W").sum(), (done["result"] == "L").sum(), (done["result"] == "P").sum()
    mp = done["model_p"].astype(float)
    exp_wins = mp.sum()
    wins_with_p = (done.loc[mp.notna(), "result"] == "W").sum()
    lines += [
        f"Record: {w}-{l}-{p}   Staked: ${staked:,.2f}   Profit: ${profit:,.2f}   "
        f"ROI: {profit / staked:.1%}" if staked else "",
        (f"Model expected {exp_wins:.1f} wins on the {mp.notna().sum()} bets it priced, "
         f"got {wins_with_p} (large gaps over 100+ bets mean the model is miscalibrated)")
        if mp.notna().any() else None,
    ]
    c = done.dropna(subset=["close_odds"])
    if len(c):
        clv = (c["close_odds"].map(implied_prob) - c["odds"].map(implied_prob))
        lines.append(f"Closing line value: {clv.mean():+.1%} avg implied prob over {len(c)} bets; "
                     f"beat the close on {(clv > 0).mean():.0%}")
        lines.append("  (positive CLV over a few hundred bets is the best sign of a real edge)")
    else:
        lines.append("Closing line value: no closing odds recorded "
                     "(use `q1.py close` - it's the most reliable test of the model)")
    done["market"] = done["period"].map({"q1": "Q1", "full": "Game"}).fillna("Q1") + " " + done["stat"]
    by = done.groupby("market").agg(bets=("bet_id", "size"), profit=("profit", "sum"),
                                    staked=("stake", "sum"))
    by["roi"] = (by["profit"] / by["staked"]).map("{:.1%}".format)
    lines += ["", "By market:", by.to_string()]
    if len(done) < 200:
        lines.append(f"\nNote: {len(done)} bets is a small sample; "
                     "results this early are mostly luck either way.")
    return "\n".join(x for x in lines if x is not None)


def delete_bet(path: Path, bet_id: int):
    df = load(path)
    save(df[df["bet_id"] != bet_id], path)
