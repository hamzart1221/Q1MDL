"""Projections for first-quarter ("q1") and full-game ("full") player props.

    mean = minutes x rate x matchup x head-to-head x back-to-back  (x injury boost)

  MINUTES   Blend of his season average and his LAST 10 GAMES (recent role
            matters most for minutes). Teammates ruled out tonight hand their
            minutes to the remaining rotation, mostly to players at the same
            position group.

  RATE      Production per minute: season rate blended with his last-10 rate,
            both shrunk toward a sensible prior (league average for full games;
            his own full-game rate for Q1).

  MATCHUP   How much of that stat the opponent allows per minute to his
            position (PG / SG / SF / PF / C), vs league. Shrunk toward neutral
            until the opponent has enough games.

  HEAD-TO-HEAD  How he has done vs this opponent this season beyond what the
            matchup explains. Shrunk toward 1.0 by number of meetings.

  BACK-TO-BACK  Teams on the second night of a back-to-back produce a bit less;
            facing a tired opponent helps a bit. Both effects are measured from
            the data and shrunk toward typical values.

  INJURY BOOST  When teammates who were playing are ruled out, the production
            they leave behind is partly absorbed by everyone else.

The count is modelled as negative binomial (Poisson with extra spread) to turn
the mean into P(over) / P(under).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from . import config, positions

ALL_STATS = config.BASE_STATS + list(config.COMBOS)
PERIODS = ("q1", "full")
PERIOD_LABELS = {"q1": "1st quarter", "full": "Full game"}
OUT_STATUSES = {"OUT", "DOUBTFUL", "SUSPENDED", "INACTIVE"}


def components(stat: str) -> list[str]:
    stat = stat.lower()
    if stat in config.COMBOS:
        return config.COMBOS[stat]
    if stat in config.BASE_STATS:
        return [stat]
    raise ValueError(f"Unknown stat '{stat}'. Use one of: {', '.join(ALL_STATS)}")


def check_period(period) -> str:
    p = str(period or "q1").strip().lower()
    p = {"1q": "q1", "first quarter": "q1", "1st quarter": "q1", "game": "full",
         "full game": "full", "fg": "full"}.get(p, p)
    if p not in PERIODS:
        raise ValueError(f"Period must be q1 or full (got {period})")
    return p


def add_combos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, parts in config.COMBOS.items():
        df[name] = df[parts].sum(axis=1)
    return df


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return " ".join(s.split())


def team_game_dates(*tables) -> set[tuple[str, pd.Timestamp]]:
    out = set()
    for t in tables:
        if t is not None and len(t):
            out |= set(zip(t["team"], t["game_date"]))
    return out


def mark_b2b(df: pd.DataFrame, dates: set) -> pd.DataFrame:
    """Add b2b (his team played yesterday) and opp_b2b columns."""
    df = df.copy()
    prev = df["game_date"] - pd.Timedelta(days=1)
    df["b2b"] = [(t, d) in dates for t, d in zip(df["team"], prev)]
    df["opp_b2b"] = [(t, d) in dates for t, d in zip(df["opp"], prev)]
    return df


def _fit_dispersion(h: pd.DataFrame | None) -> dict[str, float]:
    """Extra spread beyond Poisson: Var = mean + alpha * mean^2."""
    if h is None or h.empty:
        return {s: 0.15 for s in ALL_STATS}
    reg = h[h["starter"]]
    counts = reg.groupby(["season", "player_id"])["min"].transform("size")
    reg = reg[counts >= 15]
    alpha = {}
    for s in ALL_STATS:
        g = reg.groupby(["season", "player_id"])[s].agg(["mean", "var"])
        g = g[g["mean"] > 0.3]
        alpha[s] = (float(((g["var"] - g["mean"]) / g["mean"] ** 2).clip(lower=0.005).median())
                    if len(g) >= 5 else 0.15)
    return alpha


def _fit_b2b(h: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    """Measured effect of own / opponent back-to-back on each stat, shrunk to priors."""
    out = {"own": {}, "opp": {}}
    for kind, col, prior in [("own", "b2b", config.B2B_PRIOR_OWN),
                             ("opp", "opp_b2b", config.B2B_PRIOR_OPP)]:
        for s in config.BASE_STATS:
            out[kind][s] = prior
        if h is None or h.empty or col not in h:
            continue
        rested = h[~h["b2b"] & ~h["opp_b2b"]]
        g = rested.groupby(["season", "player_id"])
        base = g[config.BASE_STATS].mean()
        base = base[g.size() >= 8]
        tired = h[h[col]].merge(base, left_on=["season", "player_id"], right_index=True,
                                suffixes=("", "_exp"))
        n = len(tired)
        for s in config.BASE_STATS:
            exp = tired[f"{s}_exp"].sum()
            if n == 0 or exp <= 0:
                continue
            ratio = tired[s].sum() / exp
            out[kind][s] = prior + n / (n + config.B2B_SHRINK_ROWS) * (ratio - prior)
    return out


def _matchup(h: pd.DataFrame, league_rate: dict, k: float, prior: tuple | None = None):
    """Opponent per-minute allowed vs league, overall ('team') and by position ('pos')."""
    recent = (h[["opp", "game_id", "game_date"]].drop_duplicates()
              .assign(r=lambda d: d.groupby("opp")["game_date"].rank(method="first", ascending=False)))
    recent = recent[recent["r"] <= config.OPP_LOOKBACK_GAMES]
    hh = h.merge(recent[["opp", "game_id"]], on=["opp", "game_id"])

    g = hh.groupby("opp")
    n = g["game_id"].nunique()
    team = pd.DataFrame({"games": n})
    for s in config.BASE_STATS:
        raw = ((g[s].sum() / g["min"].sum()) / league_rate[s]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        base = prior[0][s].reindex(raw.index).fillna(1.0) if prior else 1.0
        team[s] = base + n / (n + k) * (raw - base)

    gp = hh.groupby(["opp", "pos5"])
    n_p = gp["game_id"].nunique()
    lg = hh.groupby("pos5")
    pos = pd.DataFrame({"games": n_p})
    for s in config.BASE_STATS:
        league_pos = lg[s].sum() / lg["min"].sum()
        lr = n_p.index.get_level_values("pos5").map(league_pos).values
        raw = ((gp[s].sum() / gp["min"].sum()) / lr).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        if prior:
            base = prior[1][s].reindex(raw.index)
            base = base.fillna(pd.Series(raw.index.get_level_values("opp").map(team[s]).values,
                                         index=raw.index)).fillna(1.0)
        else:
            base = 1.0
        pos[s] = base + n_p / (n_p + k) * (raw - base)
    return team, pos


@dataclass
class Context:
    """Tonight's situation: date, injury statuses (player_id -> status)."""
    date: pd.Timestamp
    statuses: dict[int, str] = field(default_factory=dict)
    notes: dict[int, str] = field(default_factory=dict)
    _team_adj: dict = field(default_factory=dict)

    def status(self, pid: int) -> str:
        return self.statuses.get(int(pid), "")


class Model:
    def __init__(self, q1_history: pd.DataFrame, full_history: pd.DataFrame | None = None,
                 season: str | None = None, use_full: bool = True, use_matchup: bool = True,
                 use_h2h: bool = True, use_recent: bool = True, use_b2b: bool = True,
                 pos_overrides: dict[int, str] | None = None):
        """Histories must contain only games BEFORE the ones being projected."""
        self.use_matchup, self.use_h2h = use_matchup, use_h2h
        self.use_recent, self.use_b2b = use_recent, use_b2b
        q1_all = add_combos(q1_history)
        full_all = add_combos(full_history) if full_history is not None and len(full_history) else None
        self.season = season or q1_all["season"].max()
        self.dates = team_game_dates(q1_all, full_all)
        q1_all = mark_b2b(q1_all, self.dates)
        full_all = mark_b2b(full_all, self.dates) if full_all is not None else None

        q1 = q1_all[q1_all["season"] == self.season].copy()
        full = full_all[full_all["season"] == self.season].copy() if full_all is not None else None
        if full is not None and full.empty:
            full = None
        if q1.empty and full is None:
            raise ValueError(f"No games for season {self.season}")

        self.primary_pos, game_pos = positions.assign(q1, full, pos_overrides)
        q1["pos5"] = positions.label(q1, self.primary_pos, game_pos)
        if full is not None:
            full["pos5"] = positions.label(full, self.primary_pos, game_pos)

        # stats a period has no data for yet (e.g. steals/blocks before re-download)
        self.missing = {}
        for p, h in (("q1", q1), ("full", full)):
            miss = set()
            if h is not None:
                for s in config.BASE_STATS:
                    if s not in h or h[s].isna().all():
                        miss.add(s)
                    h[s] = h[s].fillna(0) if s in h else 0
                for name, parts in config.COMBOS.items():
                    h[name] = h[parts].sum(axis=1)
                    if miss & set(parts):
                        miss.add(name)
            self.missing[p] = miss
        for t in (q1_all, full_all):
            if t is not None:
                for s in config.BASE_STATS:
                    t[s] = t[s].fillna(0) if s in t else 0

        self.has_full = full is not None
        self.q1_uses_full = use_full and self.has_full
        self.hist = {"q1": q1, "full": full}
        self.alpha = {"q1": _fit_dispersion(q1_all), "full": _fit_dispersion(full_all)}
        self.b2b = {"q1": _fit_b2b(q1_all), "full": _fit_b2b(full_all)}
        self.league = {p: ({s: max(h[s].sum() / h["min"].sum(), 1e-6) for s in ALL_STATS}
                           if h is not None and not h.empty else None)
                       for p, h in self.hist.items()}

        self.players = {}
        if full is not None:
            self.players["full"] = self._players(full, "full")
        if not q1.empty:
            self.players["q1"] = self._players(q1, "q1")

        self.matchup = {}
        if full is not None:
            self.matchup["full"] = _matchup(full, self.league["full"], config.OPP_SHRINK_GAMES["full"])
        if not q1.empty:
            prior = self.matchup.get("full") if self.q1_uses_full else None
            self.matchup["q1"] = _matchup(q1, self.league["q1"], config.OPP_SHRINK_GAMES["q1"], prior)

        self.h2h = {p: self._h2h(p) for p in self.players}

        names = pd.concat([t[["player_name", "team", "last_game", "games"]]
                           for t in self.players.values()])
        names = names.sort_values("last_game").groupby(level=0).last()
        names["name_key"] = names["player_name"].map(norm_name)
        names["pos"] = self.primary_pos.reindex(names.index).fillna("SF")
        self.directory = names

    # ---------------------------------------------------------------- tables
    def _players(self, h: pd.DataFrame, period: str) -> pd.DataFrame:
        h = h.sort_values("game_date")
        g = h.groupby("player_id")
        last10 = h.groupby("player_id").tail(config.RECENT_GAMES)
        g10 = last10.groupby("player_id")
        out = pd.DataFrame({
            "player_name": g["player_name"].last(),
            "team": g["team"].last(),
            "games": g.size(),
            "last_game": g["game_date"].max(),
            "min_avg": g["min"].mean(),
            "min_last10": g10["min"].mean(),
            "min_last5": g["min"].apply(lambda m: m.tail(5).mean()),
            "start_rate_last10": g10["starter"].mean(),
        })
        out["pos"] = self.primary_pos.reindex(out.index).fillna("SF")
        wm = config.RECENT_WEIGHT_MINUTES if self.use_recent else 0.0
        out["min_proj"] = (1 - wm) * out["min_avg"] + wm * out["min_last10"]

        tot_min, min10 = g["min"].sum(), g10["min"].sum()
        lg = self.league[period]
        anchored = period == "q1" and self.q1_uses_full
        k = config.ANCHOR_MINUTES_Q1 if anchored else config.PRIOR_MINUTES[period]
        w = config.RECENT_WEIGHT_RATE if self.use_recent else 0.0
        for s in config.BASE_STATS:
            out[f"raw_{s}"] = g[s].sum() / tot_min
            if anchored:
                f = self.players["full"]
                scale = lg[s] / self.league["full"][s]
                prior = (f[f"rate_{s}"].reindex(out.index) * scale).fillna(lg[s])
            else:
                prior = lg[s]
            season_rate = (g[s].sum() + prior * k) / (tot_min + k)
            # last-10 rate, shrunk toward his season rate
            recent_rate = ((g10[s].sum() + season_rate * config.RECENT_PRIOR_MINUTES[period])
                           / (min10 + config.RECENT_PRIOR_MINUTES[period]))
            out[f"season_rate_{s}"] = season_rate
            out[f"rate_{s}"] = (1 - w) * season_rate + w * recent_rate
            out[f"avg_{s}"] = g[s].mean()
            out[f"last10_{s}"] = g10[s].mean()
        out["prior_games"] = k / out["min_avg"].clip(lower=1)
        return out

    def _h2h(self, period: str) -> pd.DataFrame:
        src = "full" if (self.has_full and (period == "full" or self.q1_uses_full)) else "q1"
        h, players = self.hist[src], self.players[src]
        team_f, pos_f = self.matchup[src]
        g = h.groupby(["player_id", "opp"])
        out = pd.DataFrame({"h2h_games": g.size(), "h2h_min": g["min"].sum()})
        pid = out.index.get_level_values("player_id")
        opp = out.index.get_level_values("opp")
        pos = players["pos"].reindex(pid).values
        k = config.H2H_SHRINK_GAMES[src]
        w = out["h2h_games"] / (out["h2h_games"] + k)
        for s in config.BASE_STATS:
            vs_rate = g[s].sum() / out["h2h_min"]
            season_raw = pd.Series(players[f"raw_{s}"].reindex(pid).values, index=out.index)
            mf = pd.Series(pos_f[s].reindex(pd.MultiIndex.from_arrays([opp, pos])).values, index=out.index)
            mf = mf.fillna(pd.Series(team_f[s].reindex(opp).values, index=out.index)).fillna(1.0)
            resid = (vs_rate / season_raw.where(season_raw > 0) / mf).fillna(1.0).clip(0.4, 2.5)
            out[f"h2h_{s}"] = 1 + w * (resid - 1)
        out["h2h_source"] = src
        return out

    # ---------------------------------------------------------------- lookup
    def find_player(self, name_or_id) -> int:
        d = self.directory
        if isinstance(name_or_id, (int, np.integer)) or str(name_or_id).isdigit():
            pid = int(name_or_id)
            if pid in d.index:
                return pid
            raise KeyError(f"player id {pid} has no games this season")
        key = norm_name(name_or_id)
        hits = d.index[d["name_key"] == key]
        if len(hits) == 0:
            parts = key.split()
            if parts:
                cand = d[d["name_key"].str.endswith(" " + parts[-1])]
                if len(parts) > 1:
                    cand = cand[cand["name_key"].str[0] == parts[0][0]]
                if len(cand) == 1:
                    return int(cand.index[0])
            raise KeyError(f"no player matching '{name_or_id}' this season")
        if len(hits) > 1:
            return int(d.loc[hits, "last_game"].idxmax())
        return int(hits[0])

    def player_names(self, min_games: int = config.MIN_GAMES) -> list[str]:
        return sorted(self.directory.loc[self.directory["games"] >= min_games, "player_name"])

    def matchup_factor(self, period: str, opp: str, pos: str, stat: str) -> float:
        if not self.use_matchup or period not in self.matchup:
            return 1.0
        team, posf = self.matchup[period]
        if pos and (opp, pos) in posf.index:
            return float(posf.loc[(opp, pos), stat])
        if opp in team.index:
            return float(team.loc[opp, stat])
        return 1.0

    def is_b2b(self, team: str, date) -> bool:
        return (team, pd.Timestamp(date) - pd.Timedelta(days=1)) in self.dates

    def last_games(self, player, period: str = "full", n: int = config.RECENT_GAMES) -> pd.DataFrame:
        period = check_period(period)
        pid = self.find_player(player)
        h = self.hist[period]
        if h is None:
            return pd.DataFrame()
        g = add_combos(h[h["player_id"] == pid]).sort_values("game_date").tail(n)
        cols = ["game_date", "team", "opp", "home", "starter", "pos5", "min", "pts", "reb",
                "ast", "fg3m", "stl", "blk", "pra", "sb", "b2b"]
        return g[[c for c in cols if c in g]].iloc[::-1]

    # ---------------------------------------------------------------- injuries
    def team_rotation(self, period: str, team: str) -> pd.DataFrame:
        tbl = self.players[period]
        h = self.hist[period]
        tdates = sorted(h.loc[h["team"] == team, "game_date"].unique())
        if not tdates:
            return tbl.iloc[0:0]
        cutoff = tdates[-min(config.RECENT_GAMES, len(tdates))]
        rot = tbl[(tbl["team"] == team) & (tbl["last_game"] >= cutoff)
                  & (tbl["min_proj"] >= config.ROTATION_MIN[period])].copy()
        recent_cut = tdates[-min(3, len(tdates))]
        rot["recently_playing"] = rot["last_game"] >= recent_cut
        return rot

    def team_adjustments(self, period: str, team: str, ctx: Context) -> dict:
        """Minutes and rate multipliers for a team given who is ruled out tonight."""
        key = (period, team)
        if key in ctx._team_adj:
            return ctx._team_adj[key]
        rot = self.team_rotation(period, team)
        out_mask = rot.index.map(lambda p: ctx.status(p) in OUT_STATUSES).astype(bool)
        newly_out = rot[out_mask & rot["recently_playing"]]
        remaining = rot[~out_mask].copy()
        res = {"minutes": {}, "rate_mult": {}, "out_names": list(rot.loc[out_mask, "player_name"]),
               "absorbed_from": list(newly_out["player_name"])}
        if newly_out.empty or remaining.empty:
            ctx._team_adj[key] = res
            return res
        cap = config.MAX_MINUTES[period]
        new_min = remaining["min_proj"].copy()
        remaining = remaining.copy()
        for _, o in newly_out.iterrows():
            same = remaining["pos"].map(positions.GROUP) == positions.GROUP.get(o["pos"], "F")
            # players with more room to grow (bench) soak up more, same position most
            room = (cap - remaining["min_proj"]).clip(lower=0)
            wts = room * np.where(same, config.SAME_POSITION_WEIGHT, 1.0)
            if wts.sum() <= 0:
                continue
            new_min = new_min + o["min_proj"] * wts / wts.sum()
        new_min = new_min.clip(upper=cap)
        for s in config.BASE_STATS:
            before = (rot["min_proj"] * rot[f"rate_{s}"]).sum()
            after = (new_min * remaining[f"rate_{s}"]).sum()
            lost = max(before - after, 0.0)
            bump = 1 + config.USAGE_ABSORB[s] * lost / after if after > 0 else 1.0
            for pid in remaining.index:
                res["rate_mult"].setdefault(pid, {})[s] = bump
        res["minutes"] = new_min.to_dict()
        ctx._team_adj[key] = res
        return res

    # ---------------------------------------------------------------- project
    def project(self, player, opp: str, stat: str, period: str = "q1",
                minutes: float | None = None, ctx: Context | None = None,
                b2b: bool | None = None, opp_b2b: bool | None = None) -> dict:
        period = check_period(period)
        if period not in self.players:
            raise KeyError("No full-game stats downloaded yet - run Update stats."
                           if period == "full" else "No first-quarter stats yet.")
        pid = self.find_player(player)
        tbl = self.players[period]
        if pid not in tbl.index:
            raise KeyError(f"{self.directory.loc[pid, 'player_name']} has no "
                           f"{PERIOD_LABELS[period].lower()} minutes this season")
        p = tbl.loc[pid]
        opp = str(opp).upper()
        stat = stat.lower()
        if stat in self.missing.get(period, set()):
            raise KeyError(f"No {stat} data for the {PERIOD_LABELS[period].lower()} yet - "
                           "click Update stats (it re-downloads old box scores once).")

        status, adj = "", None
        if ctx is not None:
            status = ctx.status(pid)
            adj = self.team_adjustments(period, p["team"], ctx)
            if b2b is None:
                b2b = self.is_b2b(p["team"], ctx.date)
            if opp_b2b is None:
                opp_b2b = self.is_b2b(opp, ctx.date)
        b2b, opp_b2b = bool(b2b), bool(opp_b2b)

        base_min = float(p["min_proj"])
        inj_min = adj["minutes"].get(pid, base_min) if adj else base_min
        mins = float(minutes) if minutes is not None and not pd.isna(minutes) else inj_min

        h2h = self.h2h[period]
        hrow = h2h.loc[(pid, opp)] if (pid, opp) in h2h.index else None
        bf = self.b2b[period]
        mean = season_part = form_part = 0.0
        mfs, hfs, bfs, ufs = [], [], [], []
        for s in components(stat):
            mf = self.matchup_factor(period, opp, p["pos"], s)
            hf = float(hrow[f"h2h_{s}"]) if (hrow is not None and self.use_h2h) else 1.0
            b = 1.0
            if self.use_b2b:
                b *= bf["own"][s] if b2b else 1.0
                b *= bf["opp"][s] if opp_b2b else 1.0
            u = adj["rate_mult"].get(pid, {}).get(s, 1.0) if adj else 1.0
            part = mins * p[f"rate_{s}"] * mf * hf * b * u
            mean += part
            season_part += p[f"season_rate_{s}"]
            form_part += p[f"rate_{s}"]
            mfs.append(mf); hfs.append(hf); bfs.append(b); ufs.append(u)
        if status in OUT_STATUSES:
            mean = 0.0

        hh = self.hist[period]
        vs = hh[(hh["player_id"] == pid) & (hh["opp"] == opp)]
        vs = add_combos(vs) if len(vs) and stat in config.COMBOS else vs
        n_h2h = int(hrow["h2h_games"]) if hrow is not None else 0
        teammates_out = [n for n in (adj["out_names"] if adj else []) if n != p["player_name"]]
        return {
            "player_id": pid, "player_name": p["player_name"], "team": p["team"],
            "pos": p["pos"], "opp": opp, "stat": stat, "period": period, "mean": mean,
            "minutes": mins, "min_season": float(p["min_avg"]),
            "min_last10": float(p["min_last10"]), "min_last5": float(p["min_last5"]),
            "season_avg": float(sum(p[f"avg_{s}"] for s in components(stat))),
            "last10_avg": float(sum(p[f"last10_{s}"] for s in components(stat))),
            "vs_games": len(vs), "vs_avg": float(vs[stat].mean()) if len(vs) else np.nan,
            "h2h_games": n_h2h, "h2h_factor": float(np.mean(hfs)),
            "h2h_source": hrow["h2h_source"] if hrow is not None else "",
            "matchup_factor": float(np.mean(mfs)),
            "form_factor": form_part / season_part if season_part > 0 else 1.0,
            "minutes_factor": mins / float(p["min_avg"]) if p["min_avg"] > 0 else 1.0,
            "b2b": b2b, "opp_b2b": opp_b2b, "b2b_factor": float(np.mean(bfs)),
            "status": status, "injury_boost": float(np.mean(ufs)),
            "minutes_boost": mins - base_min if adj and minutes is None else 0.0,
            "teammates_out": teammates_out,
            "games": int(p["games"]), "start_rate_last10": float(p["start_rate_last10"]),
            "alpha": effective_alpha(self.alpha[period][stat], max(mean, 0.05),
                                     p["games"] + p["prior_games"]),
        }

    def project_many(self, targets: pd.DataFrame, period: str = "q1") -> pd.DataFrame:
        """Vectorised version for the backtest. targets: player_id, opp, [b2b, opp_b2b]."""
        period = check_period(period)
        t = targets.merge(self.players[period], left_on="player_id", right_index=True, how="inner")
        t = t[t["games"] >= config.MIN_GAMES].copy()
        t = t.merge(self.h2h[period], left_on=["player_id", "opp"], right_index=True, how="left")
        team, posf = self.matchup[period]
        key = pd.MultiIndex.from_arrays([t["opp"], t["pos"]])
        bf = self.b2b[period]
        own = t["b2b"].astype(bool) if "b2b" in t else pd.Series(False, index=t.index)
        oppb = t["opp_b2b"].astype(bool) if "opp_b2b" in t else pd.Series(False, index=t.index)
        for s in config.BASE_STATS:
            mf = pd.Series(posf[s].reindex(key).values, index=t.index)
            mf = mf.fillna(t["opp"].map(team[s])).fillna(1.0)
            if not self.use_matchup:
                mf = mf * 0 + 1.0
            hf = t[f"h2h_{s}"].fillna(1.0) if self.use_h2h else 1.0
            b = 1.0
            if self.use_b2b:
                b = np.where(own, bf["own"][s], 1.0) * np.where(oppb, bf["opp"][s], 1.0)
            t[f"proj_{s}"] = t["min_proj"] * t[f"rate_{s}"] * mf * hf * b
        for name, parts in config.COMBOS.items():
            t[f"proj_{name}"] = t[[f"proj_{x}" for x in parts]].sum(axis=1)
        t["n_eff"] = t["games"] + t["prior_games"]
        return t


Q1Model = Model  # old name


# --------------------------------------------------------------------------- probabilities
def effective_alpha(alpha: float, mean: float, n_eff: float) -> float:
    """Widen the distribution for uncertainty in the projection itself."""
    n = max(n_eff, 1.0)
    return alpha * (1 + 1 / n) + 1 / (n * max(mean, 1e-6))


def nb_params(mean, alpha):
    mean = np.maximum(mean, 1e-6)
    n = 1.0 / max(alpha, 1e-6)
    return n, n / (n + mean)


def prob_over_under(mean: float, alpha: float, line: float) -> tuple[float, float, float]:
    """P(over), P(under), P(push) for a count distributed NB(mean, alpha)."""
    n, p = nb_params(mean, alpha)
    if float(line).is_integer():
        k = int(line)
        push = float(nbinom.pmf(k, n, p))
        under = float(nbinom.cdf(k - 1, n, p))
    else:
        push = 0.0
        under = float(nbinom.cdf(int(np.floor(line)), n, p))
    return 1 - under - push, under, push
