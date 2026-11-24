"""Monte Carlo simulation of games and parlays.

Each simulated game:
  1. Every rotation player who isn't ruled out gets his projected points (with the
     opponent's defense vs his position, injuries to teammates, back-to-backs, form).
     Players outside the rotation add their usual bench points.
  2. Home court adds a little to the home team (measured from this season's games,
     pulled toward a typical value).
  3. A shared "pace" swing (both teams up or down together) and a team-specific
     "shooting night" swing are drawn, sized so simulated scores vary as much as
     real NBA scores do.
  4. Players' stat lines are drawn around their projections and scaled so they add
     up to the team's simulated score, so a player's big night goes with his
     team's big night (that's what makes same-game parlay legs correlated).
  5. Ties go to overtime (5 more minutes, about 10 points each) until someone wins.

The moneyline result is the share of simulations each team wins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .model import OUT_STATUSES, Context, Model, components, effective_alpha, nb_params


def american(p: float) -> str:
    """Fair American odds for a win probability."""
    if p <= 0:
        return "n/a"
    if p >= 1:
        return "-inf"
    return f"{-100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"


def implied(odds: float) -> float:
    o = float(odds)
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def payout(odds: float) -> float:
    """Profit per $1 at American odds."""
    o = float(odds)
    return o / 100 if o > 0 else 100 / -o


def margin_of_error(p: float, n: int) -> float:
    """~95% margin of error of a simulated probability."""
    return 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


@dataclass
class TeamPlan:
    team: str
    opp: str
    home: bool
    players: pd.DataFrame          # rotation players playing tonight: pid, name, pts mean, alpha
    bench_pts: float
    mean_pts: float
    out: list[str] = field(default_factory=list)


class Simulator:
    def __init__(self, model: Model, ctx: Context, seed: int | None = None):
        if "full" not in model.players:
            raise RuntimeError("Simulations need full-game stats - click Update stats.")
        self.m, self.ctx = model, ctx
        self.rng = np.random.default_rng(seed)
        self.hca = self._home_court()
        self._plans: dict[tuple[str, str], tuple[TeamPlan, TeamPlan]] = {}

    # ---------------------------------------------------------------- setup
    def _home_court(self) -> float:
        h = self.m.hist["full"]
        g = h.groupby(["game_id", "home"])["pts"].sum().unstack()
        if True not in g or False not in g:
            return config.HOME_COURT_POINTS
        diff = (g[True] - g[False]).dropna()
        n = len(diff)
        return float(config.HOME_COURT_POINTS + n / (n + config.HOME_COURT_SHRINK_GAMES)
                     * (diff.mean() - config.HOME_COURT_POINTS))

    def _team_plan(self, team: str, opp: str, home: bool) -> TeamPlan:
        m, ctx = self.m, self.ctx
        h = m.hist["full"]
        rot = m.team_rotation("full", team)
        tg = h[h["team"] == team]
        n_games = tg["game_id"].nunique()
        if n_games == 0:
            raise KeyError(f"No games for {team} this season")
        per_game = tg.groupby("game_id")[["pts", "min"]].sum()
        team_avg, team_min = per_game["pts"].mean(), per_game["min"].mean()
        # what rotation players normally contribute per team game; the rest is bench
        rg = (h[h["player_id"].isin(rot.index) & (h["team"] == team)]
              .groupby("game_id")[["pts", "min"]].sum().reindex(per_game.index, fill_value=0))
        bench = max(team_avg - rg["pts"].mean(), 0.0)
        bench_min = max(team_min - rg["min"].mean(), 0.0)
        # opponent's overall points allowed vs league, for the bench share
        allowed = h.groupby(["opp", "game_id"])["pts"].sum().groupby(level="opp").mean()
        league = allowed.mean()
        n_opp = h.loc[h["opp"] == opp, "game_id"].nunique()
        opp_f = 1 + n_opp / (n_opp + config.OPP_SHRINK_GAMES["full"]) * (allowed.get(opp, league) / league - 1)

        rows, out = [], []
        for pid in rot.index:
            st = ctx.status(pid)
            if st in OUT_STATUSES:
                out.append(rot.loc[pid, "player_name"])
                continue
            r = m.project(int(pid), opp, "pts", "full", ctx=ctx)
            rows.append({"pid": int(pid), "name": r["player_name"], "pts": r["mean"],
                         "min": r["minutes"], "alpha": r["alpha"], "status": st})
        players = pd.DataFrame(rows, columns=["pid", "name", "pts", "min", "alpha", "status"])
        # Each player's minutes come from games he played, so a full-strength rotation adds
        # up to more than a real game. Scale everyone so the team plays a normal game's minutes.
        # Trim the excess mostly from low-minute players (how coaches shorten a rotation):
        # each player gives up minutes in proportion to how far he is below 40 minutes.
        excess = players["min"].sum() + bench_min - team_min
        if excess > 0 and len(players):
            room = (config.MAX_MINUTES["full"] - players["min"]).clip(lower=1.0) ** 2
            cut = np.minimum(excess * room / room.sum(), players["min"] * 0.9)
            new_min = players["min"] - cut
            players["pts"] *= np.where(players["min"] > 0, new_min / players["min"], 1.0)
            players["min"] = new_min
        built = players["pts"].sum() + bench * opp_f
        # blend with the team's actual scoring this season vs this defense (steadies the
        # player-by-player number); injuries still move it through the player part
        baseline = team_avg * opp_f
        w = config.SIM_TEAM_BASELINE_WEIGHT
        mean = (1 - w) * built + w * baseline
        k = mean / built if built > 0 else 1.0
        players["pts"] *= k
        return TeamPlan(team, opp, home, players, bench * opp_f * k, mean, out)

    def plan(self, away: str, home: str) -> tuple[TeamPlan, TeamPlan]:
        key = (away, home)
        if key not in self._plans:
            a = self._team_plan(away, home, False)
            hm = self._team_plan(home, away, True)
            for p, sign in ((a, -1), (hm, 1)):   # split home court between the two teams
                adj = 1 + sign * (self.hca / 2) / max(p.mean_pts, 1)
                p.players["pts"] *= adj
                p.bench_pts *= adj
                p.mean_pts *= adj
            self._plans[key] = (a, hm)
        return self._plans[key]

    # ---------------------------------------------------------------- simulate
    def _nb(self, mean, alpha, n):
        nn, p = nb_params(np.asarray(mean, dtype=float), alpha)
        return self.rng.negative_binomial(nn, p, size=n)

    def simulate_game(self, away: str, home: str, n: int,
                      extra_legs: list[tuple[int, str, str]] | None = None) -> dict:
        """Returns arrays of length n: away/home scores, pace factor, and any player stats
        requested in extra_legs as (player_id, stat, period)."""
        a, hm = self.plan(away, home)
        pace = np.exp(self.rng.normal(0, config.SIM_PACE_SD, n))
        res = {"pace": pace}
        player_pts = {}
        for p, key in ((a, "away"), (hm, "home")):
            shoot = np.exp(self.rng.normal(0, config.SIM_TEAM_SD, n))
            team = p.mean_pts * pace * shoot
            team = team * np.exp(-(config.SIM_PACE_SD ** 2 + config.SIM_TEAM_SD ** 2) / 2)  # keep the mean
            # player draws, scaled to add up to the team's simulated score
            draws = np.column_stack([self._nb(r.pts, r.alpha, n) for r in p.players.itertuples()]
                                    + [self.rng.poisson(p.bench_pts, n)]) if len(p.players) else \
                self.rng.poisson(p.bench_pts, (n, 1))
            total = draws.sum(axis=1)
            scale = np.where(total > 0, team / np.maximum(total, 1), 1.0)
            scaled = draws * scale[:, None]
            res[key] = np.round(scaled.sum(axis=1))
            for j, r in enumerate(p.players.itertuples()):
                player_pts[r.pid] = np.round(scaled[:, j])
        # overtime
        away_s, home_s = res["away"].copy(), res["home"].copy()
        tied = away_s == home_s
        while tied.any():
            k = int(tied.sum())
            away_s[tied] += np.round(self.rng.normal(10.5, 3.5, k))
            home_s[tied] += np.round(self.rng.normal(10.5, 3.5, k))
            tied = away_s == home_s
        res["away"], res["home"] = away_s, home_s
        res["player_pts"] = player_pts

        # other stats for parlay legs: drawn around projections, moved with pace
        res["player_stats"] = {}
        for pid, stat, period in (extra_legs or []):
            key = (pid, stat, period)
            if period == "full" and stat == "pts" and pid in player_pts:
                res["player_stats"][key] = player_pts[pid]
                continue
            vals = np.zeros(n)
            for s in components(stat):
                if period == "full" and s == "pts" and pid in player_pts:
                    vals += player_pts[pid]
                    continue
                r = self.m.project(pid, self._opp_of(pid, away, home), s, period, ctx=self.ctx)
                if r["status"] in OUT_STATUSES:
                    continue
                # tie the stat to the team's night: pace and his team's scoring vs expectation
                team_key = "away" if self.m.directory.loc[pid, "team"] == away else "home"
                plan = a if team_key == "away" else hm
                night = np.clip(res[team_key] / max(plan.mean_pts, 1), 0.6, 1.6) ** 0.5
                vals += self._nb(r["mean"] * night, r["alpha"], n)
            res["player_stats"][key] = vals
        return res

    def _opp_of(self, pid: int, away: str, home: str) -> str:
        team = self.m.directory.loc[pid, "team"]
        return home if team == away else away

    def run_games(self, games: pd.DataFrame, n: int = 100) -> pd.DataFrame:
        rows = []
        for g in games.itertuples():
            try:
                a, hm = self.plan(g.away, g.home)
                r = self.simulate_game(g.away, g.home, n)
            except KeyError as e:
                rows.append({"game": f"{g.away} @ {g.home}", "note": str(e)})
                continue
            margin = r["home"] - r["away"]
            p_home = float((margin > 0).mean())
            tot = r["away"] + r["home"]
            rows.append({
                "game": f"{g.away} @ {g.home}", "tip": getattr(g, "tip", ""),
                "away": g.away, "home": g.home, "sims": n,
                "away_win": 1 - p_home, "home_win": p_home,
                "away_fair": american(1 - p_home), "home_fair": american(p_home),
                "away_score": float(np.mean(r["away"])), "home_score": float(np.mean(r["home"])),
                "proj_away": a.mean_pts, "proj_home": hm.mean_pts,
                "home_spread": -float(np.median(margin)), "total": float(np.median(tot)),
                "margin_sd": float(np.std(margin)), "moe": margin_of_error(p_home, n),
                "away_out": ", ".join(a.out), "home_out": ", ".join(hm.out),
                "away_range": f"{np.percentile(r['away'], 10):.0f}-{np.percentile(r['away'], 90):.0f}",
                "home_range": f"{np.percentile(r['home'], 10):.0f}-{np.percentile(r['home'], 90):.0f}",
                "note": "",
            })
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- parlays
LEG_TYPES = ("prop", "moneyline", "spread", "total")


STAT_WORDS = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "3-pointers", "stl": "steals",
              "blk": "blocks", "pra": "pts+reb+ast", "pr": "pts+reb", "pa": "pts+ast", "ra": "reb+ast",
              "sb": "steals+blocks"}


def describe_leg(leg: dict) -> str:
    t = leg["type"]
    if t == "prop":
        per = " (1st quarter)" if leg.get("period") == "q1" else ""
        return f"{leg['player']} {leg['side']} {leg['line']:g} {STAT_WORDS.get(leg['stat'], leg['stat'])}{per}"
    if t == "moneyline":
        return f"{leg['team']} moneyline"
    if t == "spread":
        return f"{leg['team']} {leg['line']:+g}"
    return f"{leg['game']} {leg['side']} {leg['line']:g}"


def simulate_parlay(model: Model, ctx: Context, games: pd.DataFrame, legs: list[dict],
                    n: int = 100, odds: float | None = None, seed: int | None = None) -> dict:
    """legs: dicts with type and fields:
        prop:      player, stat, period (full/q1), side (over/under), line
        moneyline: team
        spread:    team, line (e.g. -4.5)
        total:     team (either team in the game), side, line
    """
    sim = Simulator(model, ctx, seed)
    by_team = {}
    for g in games.itertuples():
        by_team[g.away] = (g.away, g.home)
        by_team[g.home] = (g.away, g.home)

    def game_for_team(team, who=None):
        if team not in by_team:
            label = f"{who} ({team})" if who else team
            raise ValueError(f"{label} doesn't play on the date set on Tonight's Slate. "
                             "Check the date, or remove that leg.")
        return by_team[team]

    needs: dict[tuple[str, str], list] = {}
    resolved = []
    for leg in legs:
        leg = dict(leg)
        if leg["type"] == "prop":
            pid = model.find_player(leg["player"])
            leg["player"] = str(model.directory.loc[pid, "player_name"])
            leg["pid"] = pid
            leg["team"] = str(model.directory.loc[pid, "team"])
            leg["line"] = float(leg["line"])
            leg["stat"] = leg["stat"].lower()
            leg["period"] = leg.get("period", "full")
            gm = game_for_team(leg["team"], leg["player"])
            needs.setdefault(gm, []).append((pid, leg["stat"], leg["period"]))
        else:
            leg["team"] = leg["team"].upper()
            gm = game_for_team(leg["team"])
            needs.setdefault(gm, [])
            if leg["type"] != "moneyline":
                leg["line"] = float(leg["line"])
            if leg["type"] == "total":
                leg["game"] = f"{gm[0]} @ {gm[1]}"
        leg["gm"] = gm
        resolved.append(leg)

    results = {gm: sim.simulate_game(gm[0], gm[1], n, list(dict.fromkeys(x)))
               for gm, x in needs.items()}

    hits, pushes = [], []
    for leg in resolved:
        r = results[leg["gm"]]
        away, home = leg["gm"]
        push = np.zeros(n, dtype=bool)
        if leg["type"] == "prop":
            v = r["player_stats"][(leg["pid"], leg["stat"], leg["period"])]
            hit = v > leg["line"] if leg["side"] == "over" else v < leg["line"]
            push = v == leg["line"]
        elif leg["type"] == "moneyline":
            mine, theirs = (r["away"], r["home"]) if leg["team"] == away else (r["home"], r["away"])
            hit = mine > theirs
        elif leg["type"] == "spread":
            mine, theirs = (r["away"], r["home"]) if leg["team"] == away else (r["home"], r["away"])
            adj = mine + leg["line"] - theirs
            hit, push = adj > 0, adj == 0
        else:
            tot = r["away"] + r["home"]
            hit = tot > leg["line"] if leg["side"] == "over" else tot < leg["line"]
            push = tot == leg["line"]
        hits.append(hit)
        pushes.append(push)

    hits = np.array(hits)
    pushes = np.array(pushes)
    # a pushed leg drops out of the parlay (counts as a win for that leg)
    leg_ok = hits | pushes
    all_hit = leg_ok.all(axis=0)
    p = float(all_hit.mean())
    leg_p = hits.mean(axis=1)
    independent = float(np.prod(leg_ok.mean(axis=1)))
    out = {
        "sims": n, "hits": int(all_hit.sum()), "prob": p, "moe": margin_of_error(p, n),
        "fair_odds": american(p), "independent_prob": independent,
        "legs": [{"leg": describe_leg(l), "prob": float(pr), "fair": american(float(pr)),
                  "hits": int(h.sum()), "pushes": int(ps.sum())}
                 for l, pr, h, ps in zip(resolved, leg_p, hits, pushes)],
    }
    if odds is not None and str(odds).strip():
        o = float(str(odds).replace("+", ""))
        out["book_odds"] = o
        out["book_implied"] = implied(o)
        out["ev"] = p * payout(o) - (1 - p)
    return out
