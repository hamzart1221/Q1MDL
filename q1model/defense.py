"""Defense vs position: what each team allows, per game, to PG / SG / SF / PF / C,
and its rank among the 30 teams.

Rank 1 = allows the MOST (the softest matchup, good for overs).
Rank 30 = allows the LEAST (the toughest matchup).
"""
from __future__ import annotations

import pandas as pd

from .positions import POSITIONS

STAT_NAMES = {"pts": "Points", "ast": "Assists", "reb": "Rebounds", "fg3m": "3-pointers made",
              "stl": "Steals", "blk": "Blocks"}


def allowed_per_game(h: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Average per game allowed by each team (rows) to each position (columns).
    h needs columns opp, game_id, pos5 and the stat."""
    per_game = h.groupby(["opp", "game_id", "pos5"])[stat].sum().unstack("pos5", fill_value=0)
    per_game = per_game.reindex(columns=POSITIONS, fill_value=0)
    out = per_game.groupby(level="opp").mean()
    out["games"] = per_game.groupby(level="opp").size()
    return out


def rankings(h: pd.DataFrame, stat: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(values, ranks): values = per-game allowed, ranks 1..30 with 1 = allows most."""
    vals = allowed_per_game(h, stat)
    ranks = vals[POSITIONS].rank(ascending=False, method="min").astype(int)
    return vals, ranks


def all_rankings(h: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    return {s: rankings(h, s) for s in STAT_NAMES}


def rank_for(ranks: dict, opp: str, pos5: str) -> dict[str, int | None]:
    out = {}
    for s, (_, r) in ranks.items():
        out[s] = int(r.loc[opp, pos5]) if opp in r.index and pos5 in r.columns else None
    return out
