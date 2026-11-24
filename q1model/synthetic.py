"""Fake season generator so every command can be tried without downloading.
The numbers are invented; do not bet on anything the demo shows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import COLUMNS, paths

TEAMS = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
         "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
         "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
FIRST = ["Jalen", "Marcus", "Tyrese", "Devin", "Jaylen", "Anthony", "Cam", "Darius",
         "Miles", "Evan", "Isaiah", "Malik", "Andre", "Keegan", "Scottie", "Trey"]
LAST = ["Carter", "Holmes", "Brooks", "Reed", "Walker", "Hayes", "Price", "Bennett",
        "Foster", "Grant", "Ward", "Mitchell", "Coleman", "Barnes", "Jordan", "Ellis"]
POS = ["G", "G", "F", "F", "C", "G", "F", "C", "G"]
BASE = {  # per minute by position: pts, reb, ast, fg3m, stl, blk
    "G": (0.55, 0.10, 0.17, 0.07, 0.035, 0.010), "F": (0.50, 0.17, 0.08, 0.05, 0.028, 0.018),
    "C": (0.45, 0.28, 0.07, 0.02, 0.022, 0.045)}


def generate(data_dir: str | Path, season: str = "2026-27", start: str = "2026-10-21",
             days: int = 160, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    players, pid = [], 1000
    for team in TEAMS:
        for slot in range(9):
            pos = POS[slot]
            star = rng.lognormal(0, 0.25)
            players.append(dict(
                player_id=pid, team=team, pos=pos, starter=slot < 5,
                player_name=f"{rng.choice(FIRST)} {rng.choice(LAST)} {team}{slot}",
                q1_min=rng.normal(9.3, 1.0) if slot < 5 else rng.uniform(1.5, 4.5),
                rest_min=rng.normal(25, 2.5) if slot < 5 else rng.uniform(12, 22),
                q1_boost=rng.lognormal(0, 0.08, 6),   # fast/slow starters
                rates=np.array(BASE[pos]) * star * rng.lognormal(0, 0.15, 6)))
            pid += 1
    players = pd.DataFrame(players)
    # opponent defense by position, and a small player-vs-team quirk
    defense = {(t, p): rng.lognormal(0, 0.09, 6) for t in TEAMS for p in "GFC"}
    quirk = {}

    def draw(mins, mu):
        lam = rng.gamma(4.0, mu / 4.0)      # extra spread beyond Poisson
        reb, ast, fg3m, stl, blk = rng.poisson(lam[1:])
        two = rng.poisson(max(lam[0] - 3 * fg3m, 0) / 2.3)
        ft = rng.poisson(lam[0] * 0.08)
        return dict(pts=3 * fg3m + 2 * two + ft, reb=reb, ast=ast, fg3m=fg3m,
                    fga=fg3m + two + rng.poisson(3 * mins / 10), fta=ft,
                    tov=rng.poisson(0.05 * mins), stl=stl, blk=blk)

    rows, full_rows, gid = [], [], 22600000
    last_played: dict[str, int] = {}
    for day in range(days):
        date = (pd.Timestamp(start) + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        playing = rng.permutation(TEAMS)[: 2 * rng.integers(4, 8)]
        tired = {t for t in playing if last_played.get(t) == day - 1}
        for t in playing:
            last_played[t] = day
        for home, away in zip(playing[::2], playing[1::2]):
            gid += 1
            for team, opp, is_home in [(home, away, True), (away, home, False)]:
                for _, p in players[players["team"] == team].iterrows():
                    if rng.random() < 0.06:  # rest / injury
                        continue
                    mins = float(np.clip(rng.normal(p["q1_min"], 1.3), 0, 12))
                    if mins <= 0.2:
                        continue
                    q = quirk.setdefault((p["player_id"], opp), rng.lognormal(0, 0.12, 6))
                    base = p["rates"] * defense[(opp, p["pos"])] * q
                    base = base * (0.95 if team in tired else 1.0) * (1.03 if opp in tired else 1.0)
                    q1 = draw(mins, mins * base * p["q1_boost"])
                    rest_min = float(np.clip(rng.normal(p["rest_min"], 3), 0, 36))
                    rest = draw(rest_min, rest_min * base * 0.97)
                    meta = dict(game_id=str(gid), game_date=date, season=season,
                                player_id=p["player_id"], player_name=p["player_name"],
                                team=team, opp=opp, home=is_home, starter=p["starter"],
                                pos=p["pos"] if p["starter"] else "")
                    rows.append({**meta, "min": round(mins, 2), **q1})
                    full_rows.append({**meta, "min": round(mins + rest_min, 2),
                                      **{k: q1[k] + rest[k] for k in q1}})
    df = pd.DataFrame(rows)[COLUMNS]
    p = paths(data_dir)
    p["root"].mkdir(parents=True, exist_ok=True)
    df.to_csv(p["table"], index=False)
    pd.DataFrame(full_rows)[COLUMNS].to_csv(p["full_table"], index=False)
    return df
