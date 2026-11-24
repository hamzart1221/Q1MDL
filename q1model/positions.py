"""Five positions (PG / SG / SF / PF / C) for every player.

The NBA's box scores only say G, F or C for starters, so positions are worked out:

  * Starting lineups have two guards, two forwards and a center. In each game the
    guard who hands out more assists (per minute, over the season) is the PG and
    the other the SG; the forward who grabs more rebounds is the PF, the other SF.
  * A player's position is the one he started at most often.
  * Bench players (no starts) get the position whose starters' stat profile
    (assists, rebounds, threes per minute) theirs is closest to.
  * Anything in the manual overrides (positions.csv / the app) wins.

Matchup numbers use the position a player actually started at in that game when
he started, and his usual position otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POSITIONS = ["PG", "SG", "SF", "PF", "C"]
GROUP = {"PG": "G", "SG": "G", "SF": "F", "PF": "F", "C": "C"}


def _profile(base: pd.DataFrame) -> pd.DataFrame:
    g = base.groupby("player_id")
    prof = pd.DataFrame({"min": g["min"].sum(), "ast": g["ast"].sum(), "reb": g["reb"].sum(),
                         "fg3m": g["fg3m"].sum(), "player_name": g["player_name"].last()})
    m = prof["min"].clip(lower=1)
    prof["ast_r"], prof["reb_r"], prof["fg3_r"] = prof["ast"] / m, prof["reb"] / m, prof["fg3m"] / m
    return prof


def assign(q1: pd.DataFrame | None, full: pd.DataFrame | None,
           overrides: dict[int, str] | None = None) -> tuple[pd.Series, pd.Series]:
    """Returns (primary position per player_id, position per (game_id, player_id) start)."""
    base = full if full is not None and len(full) else q1
    prof = _profile(base)

    # starters with a G/F/C slot, from either table (dedupe by game/player)
    frames = [t[t["pos"].isin(["G", "F", "C"])][["game_id", "team", "player_id", "pos"]]
              for t in (q1, full) if t is not None and len(t)]
    st = pd.concat(frames, ignore_index=True).drop_duplicates(["game_id", "player_id"])
    st = st.merge(prof[["ast_r", "reb_r"]], left_on="player_id", right_index=True, how="left")
    st = st.reset_index(drop=True)
    guards = (st["pos"] == "G").to_numpy()
    fwds = (st["pos"] == "F").to_numpy()
    # rank within each game/team: 1 = most assists (guards) / rebounds (forwards) per minute
    g_rank = st["ast_r"].where(guards).groupby([st["game_id"], st["team"]]).rank(
        method="first", ascending=False).to_numpy()
    f_rank = st["reb_r"].where(fwds).groupby([st["game_id"], st["team"]]).rank(
        method="first", ascending=False).to_numpy()
    st["pos5"] = np.select([guards & (g_rank == 1), guards, fwds & (f_rank == 1), fwds],
                           ["PG", "SG", "PF", "SF"], default="C")
    game_pos = st.set_index(["game_id", "player_id"])["pos5"]

    primary = st.groupby("player_id")["pos5"].agg(lambda s: s.mode().iat[0])

    # bench-only players: nearest starter-position profile
    feats = ["ast_r", "reb_r", "fg3_r"]
    starters = prof.loc[prof.index.intersection(primary.index)].join(primary.rename("pos5"))
    starters = starters[starters["min"] >= 60]
    if len(starters) >= 5:
        mu, sd = prof[feats].mean(), prof[feats].std().replace(0, 1)
        cent = ((starters[feats] - mu) / sd).groupby(starters["pos5"]).mean()
        rest = prof.index.difference(primary.index)
        if len(rest):
            z = (prof.loc[rest, feats] - mu) / sd
            d = {p: ((z - cent.loc[p]) ** 2).sum(axis=1) for p in cent.index}
            primary = pd.concat([primary, pd.DataFrame(d).idxmin(axis=1)])
    else:
        primary = pd.concat([primary, pd.Series("SF", index=prof.index.difference(primary.index))])

    if overrides:
        for pid, pos in overrides.items():
            if pos in POSITIONS:
                primary.loc[pid] = pos
        # an override also applies to every game he started
        mask = game_pos.index.get_level_values("player_id").isin(list(overrides))
        if mask.any():
            pids = game_pos.index.get_level_values("player_id")[mask]
            game_pos.loc[mask] = [overrides[p] for p in pids]
    return primary, game_pos


def label(table: pd.DataFrame, primary: pd.Series, game_pos: pd.Series) -> pd.Series:
    """pos5 for each row of a player-game table."""
    key = pd.MultiIndex.from_arrays([table["game_id"], table["player_id"]])
    per_game = pd.Series(game_pos.reindex(key).values, index=table.index)
    return per_game.fillna(table["player_id"].map(primary)).fillna("SF")
