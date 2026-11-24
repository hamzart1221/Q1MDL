"""Everything the desktop app does, without any window code (so it can be tested)."""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, commentary, config, data, defense, tracker
from .betting import find_edges
from .feeds import TEAM_NAMES, Injuries, fetch_schedule, team_abbr
from .model import (ALL_STATS, OUT_STATUSES, PERIOD_LABELS, Context, Model, add_combos,
                    check_period, norm_name, prob_over_under)
from .positions import POSITIONS

TEAMS = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
         "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
         "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
STAT_LABELS = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3-pointers",
               "stl": "Steals", "blk": "Blocks", "pra": "Pts+Reb+Ast", "pr": "Pts+Reb",
               "pa": "Pts+Ast", "ra": "Reb+Ast", "sb": "Steals+Blocks"}
LINE_COLS = ["period", "player", "opp", "stat", "line", "over_odds", "under_odds", "minutes"]
SLATE_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "pra", "sb"]
SPOT_CATEGORY = {"pts": "Scoring", "fg3m": "Scoring", "reb": "Rebounding", "ast": "Playmaking",
                 "stl": "Defense", "blk": "Defense", "sb": "Defense", "pra": "All-around (PRA)"}
# full-game season average needed before a stat is ranked (Q1 uses a quarter of this)
SPOT_MIN_VOLUME = {"pts": 8, "reb": 3, "ast": 2, "fg3m": 1.2, "stl": 0.8, "blk": 0.7, "sb": 1.2,
                   "pra": 12}
# 1st quarter: small counts make ratios jumpy, so the bar is higher than a quarter of the above
SPOT_MIN_VOLUME_Q1 = {"pts": 3, "reb": 1.2, "ast": 0.8, "fg3m": 0.5, "stl": 0.4, "blk": 0.4,
                      "sb": 0.6, "pra": 5}


def _tier(score: float) -> str:
    if score >= 1.15:
        return "Strong boost"
    if score >= 1.06:
        return "Favorable"
    if score > 0.95:
        return "Neutral"
    if score > 0.90:
        return "Tough"
    return "Very tough"


def default_home() -> Path:
    """Documents\\Q1 Model on Windows (works whether run as .exe or script)."""
    docs = Path(os.environ.get("USERPROFILE", Path.home())) / "Documents"
    base = docs if docs.exists() else Path.home()
    return base / "Q1 Model"


def current_season(today: date | None = None) -> str:
    t = today or date.today()
    y = t.year if t.month >= 9 else t.year - 1
    return f"{y}-{str(y + 1)[2:]}"


class Core:
    def __init__(self, home: str | Path | None = None, data_dir: str | Path | None = None):
        self.home = Path(home) if home else default_home()
        self.data_dir = Path(data_dir) if data_dir else self.home / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths = data.paths(self.data_dir)
        self.settings_path = self.home / "settings.json"
        self.lines_path = self.home / "todays_lines.csv"
        self.settings = {"bankroll": config.BANKROLL}
        if self.settings_path.exists():
            try:
                self.settings.update(json.loads(self.settings_path.read_text()))
            except (ValueError, OSError):
                pass
        self.model: Model | None = None
        self.table: pd.DataFrame | None = None
        self.full_table: pd.DataFrame | None = None
        self.lines = self._load_lines()
        self.home.mkdir(parents=True, exist_ok=True)
        self.injuries = Injuries(self.home)
        self.positions_path = self.home / "positions.csv"
        self._schedule_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------- settings
    @property
    def bankroll(self) -> float:
        return float(self.settings.get("bankroll", config.BANKROLL))

    def set_bankroll(self, value: float):
        self.settings["bankroll"] = float(value)
        self.settings_path.write_text(json.dumps(self.settings, indent=2))

    # ------------------------------------------------------------- data/model
    def has_data(self) -> bool:
        return self.paths["table"].exists()

    def load_model(self) -> str:
        if not self.has_data():
            self.model = None
            return "No stats yet. Go to the Stats tab and click Update stats."
        self.table = data.load_table(self.data_dir)
        self.full_table = data.load_full_table(self.data_dir)
        self.model = Model(self.table, self.full_table, pos_overrides=self._pos_overrides())
        last = self.table["game_date"].max().strftime("%b %d, %Y")
        full = ("full-game stats loaded" if self.model.has_full else
                "no full-game stats yet (run Update stats)")
        return (f"Season {self.model.season}: {self.table['game_id'].nunique()} games, "
                f"latest {last}; {full}")

    def player_names(self) -> list[str]:
        return self.model.player_names() if self.model is not None else []

    def player_team(self, name: str) -> str:
        if self.model is None:
            return ""
        try:
            return str(self.model.directory.loc[self.model.find_player(name), "team"])
        except KeyError:
            return ""

    def update_stats(self, progress=None, seasons=None) -> str:
        data.update(seasons or [current_season()], self.data_dir, progress=progress)
        return self.load_model()

    def make_demo(self) -> str:
        from .synthetic import generate
        generate(self.data_dir)
        return self.load_model()

    # ------------------------------------------------------------- lines
    def _load_lines(self) -> pd.DataFrame:
        if self.lines_path.exists():
            try:
                df = pd.read_csv(self.lines_path)
                if "minutes" not in df and "q1_min" in df:
                    df["minutes"] = df["q1_min"]
                df = df.reindex(columns=LINE_COLS)
                df["period"] = df["period"].fillna("q1")
                return df
            except Exception:
                pass
        return pd.DataFrame(columns=LINE_COLS)

    def save_lines(self):
        self.lines.to_csv(self.lines_path, index=False)

    def add_line(self, player, opp, stat, line, over_odds, under_odds=None, minutes=None,
                 period="q1"):
        period = check_period(period)
        stat = stat.lower()
        if stat not in ALL_STATS:
            raise ValueError(f"Stat must be one of {', '.join(ALL_STATS)}")
        line = float(line)
        over_odds = _odds(over_odds)
        under_odds = _odds(under_odds) if str(under_odds or "").strip() else None
        minutes = float(minutes) if str(minutes or "").strip() else None
        if not str(opp or "").strip():
            raise ValueError("Pick the opponent.")
        if self.model is not None:
            self.model.project(player, opp, stat, period)  # raises with a clear message
        row = dict(period=period, player=player.strip(), opp=opp.strip().upper(), stat=stat,
                   line=line, over_odds=over_odds, under_odds=under_odds, minutes=minutes)
        self.lines = pd.concat([self.lines, pd.DataFrame([row])], ignore_index=True)
        self.save_lines()

    def remove_lines(self, indexes: list[int]):
        self.lines = self.lines.drop(index=indexes).reset_index(drop=True)
        self.save_lines()

    def clear_lines(self):
        self.lines = pd.DataFrame(columns=LINE_COLS)
        self.save_lines()

    def import_lines(self, csv_path: str):
        df = pd.read_csv(csv_path)
        if "minutes" not in df and "q1_min" in df:
            df["minutes"] = df["q1_min"]
        if "period" not in df:
            df["period"] = "q1"
        df["period"] = df["period"].fillna("q1").map(check_period)
        missing = {"player", "opp", "stat", "line", "over_odds"} - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        self.lines = pd.concat([self.lines, df.reindex(columns=LINE_COLS)], ignore_index=True)
        self.save_lines()

    def edges(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Load stats first (Stats tab → Update stats).")
        if self.lines.empty:
            raise RuntimeError("Add some lines first.")
        return find_edges(self.model, self.lines, self.bankroll, self.context())

    # ------------------------------------------------------------- lookup
    # stats shown in the last-10 breakdown, with the size of a "clear" beat / miss for a full game
    BREAKDOWN = [("pts", "Pts", 5.0), ("reb", "Reb", 2.5), ("ast", "Ast", 2.0), ("fg3m", "3PM", 1.5),
                 ("stl", "Stl", 1.0), ("blk", "Blk", 1.0)]

    POS_WORDS = {"PG": "point guards", "SG": "shooting guards", "SF": "small forwards",
                 "PF": "power forwards", "C": "centers"}
    STAT_WORDS = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3-pointers",
                  "stl": "Steals", "blk": "Blocks"}

    def _last10_breakdown(self, player, lg: pd.DataFrame, period: str, ranks: dict,
                          line=None, line_stat="pts") -> list[str]:
        """Last N games in plain English: what he got, what he was expected to get against
        that defense, how tough that defense is vs his position, and whether he beat it."""
        pid = self.model.find_player(player)
        tbl = self.model.players[period]
        stats = [b for b in self.BREAKDOWN if f"avg_{b[0]}" in tbl.columns and b[0] in lg.columns]
        avg = {s: float(tbl.loc[pid, f"avg_{s}"]) for s, _, _ in stats}
        small = 0.6 if period == "q1" else 1.0

        def defense(opp, pos, s):
            """(toughness rank 1 = toughest .. 30 = easiest, expected-value factor)."""
            if not ranks or s not in ranks:
                return None, 1.0
            vals, r = ranks[s]
            if opp not in r.index or pos not in r.columns:
                return None, 1.0
            tough = len(r) + 1 - int(r.loc[opp, pos])
            lg_avg = float(vals[pos].mean())
            f = float(vals.loc[opp, pos]) / lg_avg if lg_avg > 0 else 1.0
            return tough, min(1.25, max(0.8, 1 + 0.5 * (f - 1)))   # half-weight, like the model

        def level(t):
            if t is None:
                return ""
            if t <= 5:
                return "TOUGH - top 5"
            if t <= 10:
                return "TOUGH"
            if t <= 20:
                return "AVERAGE"
            if t <= 25:
                return "EASY"
            return "EASY - bottom 5"

        def verdict(sym, t, diff, unit):
            amt = f"{max(1, round(abs(diff)))}"
            if sym == "▲":
                why = (" even vs a tough defense!" if t and t <= 10 else
                       " - took advantage of an easy matchup" if t and t >= 21 else "")
                return f"beat it by {amt}{why}"
            if sym == "▼":
                why = (" - the tough defense slowed him" if t and t <= 10 else
                       " even though it was an easy matchup" if t and t >= 21 else "")
                return f"fell short by {amt}{why}"
            return "right about what was expected"

        n = len(lg)
        main_pos = str(tbl.loc[pid, "pos"])
        out = [f"LAST {n} GAMES, GAME BY GAME ({PERIOD_LABELS[period].lower()})",
               "How to read it:",
               "  ▲ = did BETTER than expected    ▼ = did WORSE than expected    ● = about what was expected",
               "  Expected = his season average, adjusted for how good that defense is.",
               f"  Defense rank = how hard that team is on {self.POS_WORDS.get(main_pos, 'his position')} "
               "in that stat: #1 = toughest, #30 = easiest.",
               "  Season averages: " + ", ".join(f"{avg[s]:.1f} {self.STAT_WORDS[s].lower()}"
                                                 for s, _, _ in stats),
               ""]
        marks = []   # (stat, date, where, tough, diff)
        for g in lg.itertuples():
            where = ("vs " if g.home else "@ ") + g.opp
            pos = getattr(g, "pos5", None) or main_pos
            extra = "  (back-to-back)" if g.b2b else ""
            out.append(f"{g.game_date.strftime('%b %d')} {where}  -  played {g.min:.0f} min, "
                       f"{g.pra:.0f} PRA{extra}")
            for s, lab, minimum in stats:
                val = float(getattr(g, s))
                t, f = defense(g.opp, pos, s)
                exp = avg[s] * f
                diff = val - exp
                band = max(0.15 * exp, 0.3 * minimum * small)
                sym = "▲" if diff > band else "▼" if diff < -band else "●"
                if sym != "●":
                    marks.append((s, g.game_date, where, t, diff))
                rank_txt = f"{g.opp} defense #{t} of 30 ({level(t)})" if t else f"{g.opp} defense: no rank yet"
                out.append(f"   {sym} {self.STAT_WORDS[s]:<11}{val:>3.0f}   expected {exp:>4.1f}   "
                           f"{rank_txt:<39}  {verdict(sym, t, diff, lab)}")
            out.append("")
        if line is not None and line_stat in lg:
            hits = int((lg[line_stat] > float(line)).sum())
            out.append(f"He went OVER {line:g} {STAT_LABELS[line_stat].lower()} in {hits} of his last {n} games.")
            out.append("")

        # plain-English summary per stat
        out.append("THE SHORT VERSION")
        for s, lab, _ in stats:
            ups = [m for m in marks if m[0] == s and m[4] > 0]
            downs = [m for m in marks if m[0] == s and m[4] < 0]
            word = self.STAT_WORDS[s]
            fmt_m = lambda m: f"{m[1].strftime('%b %d')} {m[2]}" + (f" (defense #{m[3]})" if m[3] else "")  # noqa: E731
            line_txt = (f"  {word}: better than expected in {len(ups)} of {n} games, worse in {len(downs)}."
                        if ups or downs else f"  {word}: right about what was expected every game.")
            out.append(line_txt)
            if ups:
                out.append("     ▲ Best games: " + ", ".join(fmt_m(m) for m in sorted(ups, key=lambda m: -m[4])[:4]))
            if downs:
                out.append("     ▼ Worst games: " + ", ".join(fmt_m(m) for m in sorted(downs, key=lambda m: m[4])[:4]))

        # tough vs easy defenses
        split = []
        for s, lab, _ in stats:
            tough_g, easy_g = [], []
            for g in lg.itertuples():
                t, _ = defense(g.opp, getattr(g, "pos5", None) or main_pos, s)
                if t is None:
                    continue
                (tough_g if t <= 10 else easy_g if t >= 21 else []).append(float(getattr(g, s)))
            if tough_g and easy_g:
                a, b = sum(tough_g) / len(tough_g), sum(easy_g) / len(easy_g)
                tip = ("he does better vs easy defenses" if b - a >= max(1.0, 0.15 * avg[s]) else
                       "he does better vs tough defenses (odd, may be luck)" if a - b >= max(1.0, 0.15 * avg[s])
                       else "the defense doesn't change him much")
                split.append(f"  {self.STAT_WORDS[s]}: {a:.1f} vs tough defenses ({len(tough_g)} game{'s' * (len(tough_g) != 1)}), "
                             f"{b:.1f} vs easy ones ({len(easy_g)} game{'s' * (len(easy_g) != 1)}) - {tip}")
        if split:
            out += ["", "TOUGH vs EASY DEFENSES (last 10)"] + split
        out.append("(Defense ranks are this season's ranks as of today.)")
        return out

    def player_card(self, player: str, opp: str, period: str = "full",
                    minutes: float | None = None, day=None) -> dict:
        """Numbers for the player card at the top of Player Lookup."""
        period = check_period(period)
        opp = team_abbr(opp) or str(opp).strip().upper()
        ctx = self.context(day)
        out = {"stats": {}}
        for s in ["pts", "reb", "ast", "fg3m", "pra", "sb"]:
            if s in self.model.missing.get(period, set()):
                continue
            r = self.model.project(player, opp, s, period, minutes, ctx=ctx)
            out["stats"][s] = (r["mean"], r["season_avg"], r["last10_avg"])
            out.update(name=r["player_name"], team=r["team"], opp=r["opp"], pos=r["pos"],
                       status=r["status"], minutes=r["minutes"], b2b=r["b2b"], opp_b2b=r["opp_b2b"],
                       teammates_out=r["teammates_out"], games=r["games"], period=period)
        return out

    def lookup(self, player: str, opp: str, line: float | None = None,
               minutes: float | None = None, period: str = "full", line_stat: str = "pts",
               day=None) -> str:
        if self.model is None:
            raise RuntimeError("Load stats first (Stats tab → Update stats).")
        period = check_period(period)
        opp = team_abbr(opp) or str(opp).strip().upper()
        if not opp:
            raise ValueError("Pick the opponent.")
        ctx = self.context(day)
        try:
            ranks = self.rankings_all("full" if self.model.has_full else period)
        except RuntimeError:
            ranks = {}
        out = []
        stats = [s for s in ["pts", "reb", "ast", "fg3m", "stl", "blk", "pra", "sb"]
                 if s not in self.model.missing.get(period, set())]
        projs = {s: self.model.project(player, opp, s, period, minutes, ctx=ctx) for s in stats}
        rk_all = {}
        if ranks:
            pos0 = projs["pts"]["pos"]
            rk_all = defense.rank_for(ranks, opp, pos0)
        lg0 = self.model.last_games(player, period)
        if line_stat not in projs:
            line_stat = "pts"
        take = commentary.write(projs, rk_all, lg0, period, line, line_stat)
        for i, s in enumerate(stats):
            r = projs[s]
            if i == 0:
                out.append(f"{r['player_name']}  ({r['team']}, {r['pos']})  vs {r['opp']}"
                           f"   -   {PERIOD_LABELS[period].upper()}")
                out.append("")
                out.append("MODEL'S TAKE")
                out += ["  " + ln if ln else "" for ln in take.splitlines()]
                out.append("")
                out.append("=" * 82)
                tags = []
                if r["status"]:
                    tags.append(f"STATUS: {r['status']}")
                if r["b2b"]:
                    tags.append(f"{r['team']} on 2nd night of back-to-back")
                if r["opp_b2b"]:
                    tags.append(f"{r['opp']} on 2nd night of back-to-back")
                if r["teammates_out"]:
                    tags.append("teammates out: " + ", ".join(r["teammates_out"]))
                if tags:
                    out.append("  >> " + "  |  ".join(tags))
                out.append(f"{r['games']} games this season · started "
                           f"{r['start_rate_last10']:.0%} of last 10")
                mb = (f", +{r['minutes_boost']:.1f} from injuries" if r["minutes_boost"] > 0.05 else "")
                out.append(f"Minutes used: {r['minutes']:.1f}   (season {r['min_season']:.1f}, "
                           f"last 10 {r['min_last10']:.1f}{mb})")
                out.append("")
                out.append(f"{'Stat':<13}{'Proj':>6}{'Season':>8}{'Last10':>8}   {'vs ' + r['opp']:<15}"
                           f"{'Matchup':>8}{'Rank':>6}{'H2H':>6}{'B2B':>6}")
                out.append("-" * 82)
            vs = f"{r['vs_avg']:.1f} in {r['vs_games']}g" if r["vs_games"] else "no meetings"
            rk = defense.rank_for(ranks, opp, r["pos"]).get(s) if ranks and s in defense.STAT_NAMES else None
            out.append(f"{STAT_LABELS[s]:<13}{r['mean']:>6.1f}{r['season_avg']:>8.1f}{r['last10_avg']:>8.1f}   "
                       f"{vs:<15}{r['matchup_factor']:>8.2f}{(str(rk) if rk else '-'):>6}"
                       f"{r['h2h_factor']:>6.2f}{r['b2b_factor']:>6.2f}")
            if line is not None and s == line_stat:
                po, pu, pp = prob_over_under(r["mean"], r["alpha"], float(line))
                out.append(f"{'':13}line {line}: over {po:.0%} · under {pu:.0%}"
                           + (f" · push {pp:.0%}" if pp > 0.005 else ""))

        lg = lg0
        if len(lg):
            out += [""] + self._last10_breakdown(player, lg, period, ranks, line, line_stat)
        out += ["",
                "Rank (in the matchup table at the top) = opponent's rank in that stat allowed to his "
                "position (1 = allows the most). The game-by-game section flips it to toughness "
                "(1 = toughest).",
                "Matchup above 1.00 = opponent gives up more than average to his position.",
                "H2H = his results vs this team beyond the matchup. B2B = back-to-back effect "
                "(1.00 = rested)."]
        return "\n".join(out)

    # ------------------------------------------------------------- positions
    def position_overrides(self) -> pd.DataFrame:
        if self.positions_path.exists():
            try:
                return pd.read_csv(self.positions_path).reindex(columns=["player", "pos"])
            except Exception:  # noqa: BLE001
                pass
        return pd.DataFrame(columns=["player", "pos"])

    def _pos_overrides(self) -> dict[int, str]:
        o = self.position_overrides()
        if o.empty or self.table is None:
            return {}
        names = pd.concat([t[["player_id", "player_name"]] for t in (self.table, self.full_table)
                           if t is not None]).drop_duplicates("player_id")
        key = names.assign(k=names["player_name"].map(norm_name)).set_index("k")["player_id"]
        out = {}
        for _, r in o.iterrows():
            k = norm_name(r["player"])
            if k in key.index and str(r["pos"]).upper() in POSITIONS:
                out[int(key[k] if not isinstance(key[k], pd.Series) else key[k].iloc[-1])] = str(r["pos"]).upper()
        return out

    def set_position(self, player: str, pos: str) -> str:
        pos = str(pos).strip().upper()
        if pos not in POSITIONS:
            raise ValueError(f"Position must be one of {', '.join(POSITIONS)}")
        name = player.strip()
        if self.model is not None:
            name = str(self.model.directory.loc[self.model.find_player(player), "player_name"])
        o = self.position_overrides()
        o = o[o["player"].map(norm_name) != norm_name(name)]
        pd.concat([o, pd.DataFrame([{"player": name, "pos": pos}])]).to_csv(self.positions_path, index=False)
        return self.load_model() if self.model is not None else ""

    def clear_position(self, player: str) -> str:
        o = self.position_overrides()
        o[o["player"].map(norm_name) != norm_name(player)].to_csv(self.positions_path, index=False)
        return self.load_model() if self.model is not None else ""

    def position_of(self, player: str) -> str:
        return str(self.model.directory.loc[self.model.find_player(player), "pos"]) if self.model else ""

    # ------------------------------------------------------------- injuries
    def refresh_injuries(self) -> str:
        n = self.injuries.refresh()
        return f"Injury report updated: {n} players listed ({self.injuries.last_refresh()})"

    def injury_table(self) -> pd.DataFrame:
        df = self.injuries.combined()
        if self.model is not None and len(df):
            def mins(name):
                try:
                    pid = self.model.find_player(name)
                    p = self.model.players.get("full", self.model.players.get("q1"))
                    return round(float(p.loc[pid, "min_proj"]), 1) if pid in p.index else None
                except KeyError:
                    return None
            df["min_avg"] = df["player"].map(mins)
        return df

    def context(self, day=None) -> Context:
        """Tonight's situation: injuries mapped to players, date for back-to-backs."""
        d = pd.Timestamp(day or date.today()).normalize()
        statuses, notes = {}, {}
        if self.model is not None:
            for _, r in self.injuries.combined().iterrows():
                try:
                    pid = self.model.find_player(r["player"])
                except KeyError:
                    continue
                statuses[pid] = str(r["status"]).upper()
                notes[pid] = str(r.get("note") or "")
        return Context(d, statuses, notes)

    # ------------------------------------------------------------- schedule / slate
    def schedule(self, day=None, refresh: bool = False) -> tuple[pd.DataFrame, str]:
        d = pd.Timestamp(day or date.today()).strftime("%Y-%m-%d")
        cache = self.home / f"schedule_{d}.csv"
        if not refresh and d in self._schedule_cache:
            return self._schedule_cache[d], "cached"
        try:
            games, src = fetch_schedule(d)
            games.to_csv(cache, index=False)
        except RuntimeError:
            if cache.exists():
                games, src = pd.read_csv(cache), "saved copy"
            else:
                raise
        self._schedule_cache[d] = games
        return games, src

    def slate(self, day=None, period: str = "full", game: str | None = None,
              refresh_schedule: bool = False) -> tuple[pd.DataFrame, str]:
        """Projections for every rotation player in the day's games."""
        if self.model is None:
            raise RuntimeError("Load stats first (Stats tab → Update stats).")
        period = check_period(period)
        if period not in self.model.players:
            raise RuntimeError("No full-game stats yet - click Update stats." if period == "full"
                               else "No first-quarter stats yet.")
        games, src = self.schedule(day, refresh_schedule)
        if games.empty:
            return pd.DataFrame(), f"No games scheduled ({src})."
        ctx = self.context(day)
        ranks = self.rankings_all("full" if self.model.has_full else period)
        rows = []
        for g in games.itertuples():
            label = f"{g.away} @ {g.home}"
            if game and game not in ("All games", label):
                continue
            for team, opp, home in [(g.away, g.home, False), (g.home, g.away, True)]:
                rot = self.model.team_rotation(period, team)
                # include injured players so their status shows
                for pid in rot.sort_values("min_proj", ascending=False).index:
                    row = {"game": label, "tip": g.tip, "player": rot.loc[pid, "player_name"],
                           "team": team, "opp": opp, "home": home}
                    for s in SLATE_STATS:
                        if s in self.model.missing.get(period, set()):
                            continue
                        r = self.model.project(int(pid), opp, s, period, ctx=ctx)
                        row[s] = r["mean"]
                        row[f"l10_{s}"] = r["last10_avg"]
                        row[f"avg_{s}"] = r["season_avg"]
                        row[f"mf_{s}"] = r["matchup_factor"]
                        row[f"hf_{s}"] = r["h2h_factor"]
                        row[f"bf_{s}"] = r["b2b_factor"]
                        row[f"uf_{s}"] = r["injury_boost"]
                        row[f"ff_{s}"] = r["form_factor"]
                    r = self.model.project(int(pid), opp, "pts", period, ctx=ctx)
                    row.update(pos=r["pos"], minutes=r["minutes"], min_l10=r["min_last10"],
                               min_season=r["min_season"], status=r["status"], b2b=r["b2b"],
                               opp_b2b=r["opp_b2b"], injury_boost=r["injury_boost"],
                               minutes_boost=r["minutes_boost"], vs_games=r["vs_games"],
                               teammates_out=", ".join(r["teammates_out"][:3]),
                               starter_rate=r["start_rate_last10"])
                    for s, rk in defense.rank_for(ranks, opp, r["pos"]).items():
                        row[f"rank_{s}"] = rk
                    rows.append(row)
        return pd.DataFrame(rows), f"{len(games)} games ({src})"

    def best_spots(self, slate: pd.DataFrame, period: str = "full") -> pd.DataFrame:
        """Every player/stat tonight scored as projection / season average, with reasons."""
        if slate is None or slate.empty:
            return pd.DataFrame()
        period = check_period(period)
        scale = 0.25 if period == "q1" else 1.0
        min_minutes = 4.0 if scale < 1 else 15.0
        rows = []
        for r in slate.itertuples(index=False):
            r = r._asdict()
            if str(r["status"]) in OUT_STATUSES or r["minutes"] < min_minutes:
                continue
            for s, cat in SPOT_CATEGORY.items():
                if s not in r or pd.isna(r.get(s)):
                    continue
                avg = r[f"avg_{s}"]
                floor = SPOT_MIN_VOLUME_Q1[s] if period == "q1" else SPOT_MIN_VOLUME[s]
                if avg < floor:
                    continue
                score = r[s] / avg if avg > 0 else np.nan
                reasons = []
                mins_f = r["minutes"] / r["min_season"] if r["min_season"] else 1.0
                if r["teammates_out"] and (r["minutes_boost"] > 0.5 * scale or r[f"uf_{s}"] > 1.01):
                    reasons.append((abs(mins_f * r[f"uf_{s}"] - 1) + 0.5,
                                    f"{r['teammates_out']} out: +{r['minutes_boost']:.1f} min"
                                    + (f", usage {r[f'uf_{s}'] - 1:+.0%}" if r[f"uf_{s}"] > 1.01 else "")))
                elif abs(mins_f - 1) >= 0.05:
                    reasons.append((abs(mins_f - 1), f"minutes {'up' if mins_f > 1 else 'down'} "
                                    f"lately ({r['minutes']:.0f} vs {r['min_season']:.0f})"))
                rk = r.get(f"rank_{s}")
                mf = r[f"mf_{s}"]
                if abs(mf - 1) >= 0.03 or (rk and (rk <= 5 or rk >= 26)):
                    rk_txt = f", ranks {int(rk)}/30 vs {r['pos']}" if rk and not pd.isna(rk) else ""
                    reasons.append((abs(mf - 1), f"matchup x{mf:.2f}{rk_txt}"))
                ff = r[f"ff_{s}"]
                if abs(ff - 1) >= 0.03:
                    reasons.append((abs(ff - 1), f"{'hot' if ff > 1 else 'cold'} last 10 "
                                    f"({r[f'l10_{s}']:.1f} vs {avg:.1f})"))
                bf = r[f"bf_{s}"]
                if abs(bf - 1) >= 0.01:
                    who = "own back-to-back" if r["b2b"] and not r["opp_b2b"] else (
                        "opp on back-to-back" if r["opp_b2b"] and not r["b2b"] else "both on back-to-backs")
                    reasons.append((abs(bf - 1), f"{who} {bf - 1:+.0%}"))
                hf = r[f"hf_{s}"]
                if abs(hf - 1) >= 0.03:
                    reasons.append((abs(hf - 1), f"history vs {r['opp']} {hf - 1:+.0%}"))
                reasons.sort(reverse=True)
                rows.append({
                    # Q1: players who play most of the quarter are more reliable - weight by minutes
                    "period": period,
                    "min_weight": (float(np.clip(r["minutes"] / config.Q1_FULL_WEIGHT_MINUTES,
                                                 config.Q1_WEIGHT_RANGE[0], config.Q1_WEIGHT_RANGE[1]))
                                   if period == "q1" else 1.0),
                    "score": score, "category": cat, "stat": s,
                    "player": r["player"], "team": r["team"], "opp": r["opp"], "game": r["game"],
                    "pos": r["pos"], "proj": r[s], "season": avg, "last10": r[f"l10_{s}"],
                    "opp_rank": rk if rk and not pd.isna(rk) else None, "status": r["status"],
                    "minutes": r["minutes"], "matchup": mf, "why": "; ".join(t for _, t in reasons[:3]) or "no big factors",
                })
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["rating"] = out["score"] * out["min_weight"]
        out["tier"] = out["rating"].map(_tier)
        return out.sort_values("rating", ascending=False).reset_index(drop=True)

    def best_spots_all(self, day=None) -> tuple[pd.DataFrame, str]:
        """Best spots for both full game and 1st quarter, in one table."""
        if self.model is None:
            raise RuntimeError("Load stats first (Stats tab → Update stats).")
        frames, msgs = [], []
        for period in ("full", "q1"):
            if period not in self.model.players:
                msgs.append(f"no {PERIOD_LABELS[period].lower()} stats yet")
                continue
            sl, msg = self.slate(day, period)
            if period == "full" or not msgs:
                msgs.insert(0, msg)
            frames.append(self.best_spots(sl, period))
        frames = [f for f in frames if len(f)]
        out = (pd.concat(frames, ignore_index=True).sort_values("rating", ascending=False)
               .reset_index(drop=True) if frames else pd.DataFrame())
        return out, "; ".join(dict.fromkeys(msgs))

    # ------------------------------------------------------------- favorite picks
    FAV_STATS = ["pts", "reb", "ast", "fg3m", "pra"]
    FAV_MIN_MINUTES = 30.0
    FAV_SIMILAR_TEAMS = 5        # opponent's closest look-alikes (plus the opponent itself)
    FAV_SHRINK_MIN = 150.0       # ~5 games of minutes before similar-team history counts fully
    FAV_B2B_SHRINK_MIN = 240.0   # ~8 back-to-backs before his own B2B history counts fully

    def similar_teams(self, opp: str, pos: str, k: int | None = None) -> list[str]:
        """Teams whose defense looks most like `opp`'s against this position (pts, reb, ast, 3s
        allowed to the position + total points allowed)."""
        k = k or self.FAV_SIMILAR_TEAMS
        ranks = self.rankings_all("full")
        cols = {}
        for s in ("pts", "reb", "ast", "fg3m"):
            if s in ranks and pos in ranks[s][0].columns:
                cols[s] = ranks[s][0][pos]
        if not cols:
            return []
        prof = pd.DataFrame(cols)
        h = self.model.hist["full"]
        prof["allowed"] = h.groupby(["opp", "game_id"])["pts"].sum().groupby(level="opp").mean()
        prof = prof.dropna()
        if opp not in prof.index:
            return []
        z = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)
        d = ((z - z.loc[opp]) ** 2).sum(axis=1) ** 0.5
        return [t for t in d.sort_values().index if t != opp][:k]

    def _rate_factor(self, games: pd.DataFrame, stat: str, rate: float, shrink: float) -> tuple[float, int]:
        """His per-minute rate in these games vs his season rate, shrunk toward 1.0."""
        if games.empty or rate <= 0:
            return 1.0, 0
        val, mins = float(games[stat].sum()), float(games["min"].sum())
        return (val + rate * shrink) / (rate * (mins + shrink)), len(games)

    def favorite_picks(self, day=None, n: int = 10) -> tuple[pd.DataFrame, str]:
        """7-10 favorite full-game overs AND 7-10 favorite unders: 30+ projected minutes, a favorable matchup, his own
        back-to-back history (when it applies) and how he's done vs teams like tonight's opponent."""
        if self.model is None:
            raise RuntimeError("Load stats first (Stats tab → Update stats).")
        if not self.model.has_full:
            raise RuntimeError("Favorite Picks needs full-game stats - click Update stats.")
        sl, msg = self.slate(day, "full")
        if sl.empty:
            return pd.DataFrame(), msg
        h = self.model.hist["full"]
        h = h.assign(pra=h["pts"] + h["reb"] + h["ast"]) if "pra" not in h else h
        tbl = self.model.players["full"]
        ctx = self.context(day)
        sim_cache = {}
        cand = []
        for r in sl.itertuples(index=False):
            r = r._asdict()
            if str(r["status"]) in OUT_STATUSES or r["minutes"] <= self.FAV_MIN_MINUTES:
                continue
            try:
                pid = self.model.find_player(r["player"])
            except Exception:  # noqa: BLE001
                continue
            mine = h[h["player_id"] == pid]
            key = (r["opp"], r["pos"])
            if key not in sim_cache:
                sim_cache[key] = self.similar_teams(r["opp"], r["pos"])
            similar = sim_cache[key]
            vs_sim = mine[mine["opp"].isin(similar + [r["opp"]])]
            b2b_games = mine[mine["b2b"].astype(bool)] if "b2b" in mine else mine.iloc[0:0]
            season_min = float(mine["min"].sum()) or 1.0
            for s in self.FAV_STATS:
                if s not in r or pd.isna(r.get(s)):
                    continue
                avg = r[f"avg_{s}"]
                if avg < SPOT_MIN_VOLUME.get(s, 0):
                    continue
                rate = float(mine[s].sum()) / season_min
                sim_f, sim_n = self._rate_factor(vs_sim, s, rate, self.FAV_SHRINK_MIN)
                own_b2b, b2b_n = self._rate_factor(b2b_games, s, rate, self.FAV_B2B_SHRINK_MIN)
                sim_f = float(np.clip(sim_f, 0.85, 1.15))     # small samples: keep it a nudge
                own_b2b = float(np.clip(own_b2b, 0.90, 1.10))
                mf = r[f"mf_{s}"]
                b2b_f = own_b2b if r["b2b"] else 1.0
                # model's projection, with its league-wide B2B effect swapped for his own history,
                # nudged by his results vs look-alike opponents
                proj = r[s] / (r[f"bf_{s}"] or 1.0) * b2b_f * sim_f
                score = proj / avg if avg else np.nan
                cand.append(dict(player=r["player"], team=r["team"], opp=r["opp"], game=r["game"],
                                 pos=r["pos"], status=r["status"], stat=s, minutes=r["minutes"],
                                 proj=proj, season=avg, last10=r[f"l10_{s}"], matchup=mf,
                                 opp_rank=r.get(f"rank_{s}"), similar=similar, sim_f=sim_f,
                                 sim_n=sim_n, b2b=bool(r["b2b"]), b2b_f=own_b2b, b2b_n=b2b_n,
                                 form=r[f"ff_{s}"], teammates_out=r["teammates_out"],
                                 inj=r[f"uf_{s}"], score=score, pid=pid))
        if not cand:
            return pd.DataFrame(), msg + " - no one projected for 30+ minutes."
        c = pd.DataFrame(cand)

        c["side"] = "over"
        u = c.copy()
        u["side"] = "under"
        c["strength"] = c["score"]
        u["strength"] = 1 / u["score"].where(u["score"] > 0)

        def pick(df, limit, taken=(), per_game=None):
            df = df.sort_values("strength", ascending=False)
            df = df[~df["player"].isin(taken)].drop_duplicates("player")   # one pick per player
            out, per_game = [], per_game if per_game is not None else {}
            for row in df.itertuples():
                if len(out) >= limit:
                    break
                if per_game.get(row.game, 0) >= 3:            # spread across games
                    continue
                per_game[row.game] = per_game.get(row.game, 0) + 1
                out.append(row.Index)
            return df.loc[out]

        # OVERS: favorable matchup, good vs look-alike defenses, fine on his own back-to-backs
        good_matchup = (c["matchup"] >= 1.03) | ((c["matchup"] >= 1.01) & (c["opp_rank"].fillna(30) <= 10))
        over_strict = c[good_matchup & (c["sim_f"] >= 1.0) & (c["sim_n"] >= 2)
                        & (c["b2b_f"].where(c["b2b"], 1.0) >= 0.98) & (c["score"] >= 1.03)]
        over_loose = c[(c["matchup"] >= 0.99) & (c["score"] >= 1.0)]
        # UNDERS: the mirror image - tough matchup, struggles vs look-alike defenses, drops on B2Bs
        bad_matchup = (u["matchup"] <= 0.97) | ((u["matchup"] <= 0.99) & (u["opp_rank"].fillna(1) >= 21))
        under_strict = u[bad_matchup & (u["sim_f"] <= 1.0) & (u["sim_n"] >= 2)
                         & (u["b2b_f"].where(u["b2b"], 1.0) <= 1.02) & (u["score"] <= 0.97)]
        under_loose = u[(u["matchup"] <= 1.01) & (u["score"] <= 1.0)]

        relaxed = []

        def side_list(strict, loose, allc, name):
            per_game = {}
            got = pick(strict, n, per_game=per_game)
            if len(got) < 7:                      # always 7-10 per side: fill from the next best
                relaxed.append(name)
                more = pick(loose, 7 - len(got), taken=set(got["player"]), per_game=per_game)
                got = pd.concat([got, more])
                if len(got) < 7:
                    more = pick(allc, 7 - len(got), taken=set(got["player"]), per_game=per_game)
                    got = pd.concat([got, more])
            got = got.sort_values("strength", ascending=False).copy()
            got["rank"] = range(1, len(got) + 1)
            return got

        overs = side_list(over_strict, over_loose, c, "overs")
        unders = side_list(under_strict, under_loose, u, "unders")
        picks = pd.concat([overs, unders])
        # target line: overs - highest x.5 he still clears ~60% of the time;
        # unders - lowest x.5 he still stays under ~60% of the time
        targets, chances = [], []
        for row in picks.itertuples():
            alpha = self.model.project(int(row.pid), row.opp, row.stat, "full", ctx=ctx)["alpha"]
            best, chance = None, None
            grid = np.arange(0.5, max(row.proj * 2, 4), 1.0)
            if row.side == "over":
                for L in grid:
                    p = prob_over_under(row.proj, alpha, float(L))[0]
                    if p >= 0.60:
                        best, chance = float(L), p
                if best is None:
                    best, chance = 0.5, prob_over_under(row.proj, alpha, 0.5)[0]
            else:
                for L in grid:
                    p = prob_over_under(row.proj, alpha, float(L))[1]
                    if p >= 0.60:
                        best, chance = float(L), p
                        break
                if best is None:
                    best = float(grid[-1])
                    chance = prob_over_under(row.proj, alpha, best)[1]
            targets.append(best)
            chances.append(chance)
        picks["target"] = targets
        picks["chance"] = chances
        picks["pick"] = [f"{side.title()} {t:g} {STAT_LABELS[s].lower()}"
                         for side, t, s in zip(picks["side"], picks["target"], picks["stat"])]
        picks["why"] = [self._fav_why(r) for r in picks.to_dict("records")]
        note = msg + (f" - not enough perfect {' or '.join(relaxed)}, so a few have one weaker factor"
                      if relaxed else "")
        return picks.drop(columns=["pid"]).reset_index(drop=True), note

    def _fav_why(self, r: dict) -> str:
        bits = [f"{r['minutes']:.0f} min projected"]
        rk = r.get("opp_rank")
        rk_txt = f" (#{int(rk)} most allowed to {r['pos']}s)" if rk and not pd.isna(rk) else ""
        tone = "tough" if r["matchup"] < 0.99 else "good" if r["matchup"] > 1.01 else "neutral"
        bits.append(f"{tone} {r['opp']} matchup x{r['matchup']:.2f}{rk_txt}")
        if r["sim_n"]:
            bits.append(f"vs teams like {r['opp']} ({', '.join(r['similar'][:3])}...): "
                        f"{r['sim_f'] - 1:+.0%} in {r['sim_n']} game{'s' * (r['sim_n'] != 1)}")
        else:
            bits.append(f"no games yet vs teams like {r['opp']}")
        if r["b2b"]:
            bits.append(f"back-to-back tonight: his B2B history {r['b2b_f'] - 1:+.0%} ({r['b2b_n']} game{'s' * (r['b2b_n'] != 1)})")
        else:
            bits.append("rested (not a back-to-back)")
        if r["teammates_out"] and r["inj"] > 1.01:
            bits.append(f"{r['teammates_out']} out: usage +{r['inj'] - 1:.0%}")
        if abs(r["form"] - 1) >= 0.04:
            bits.append(f"{'hot' if r['form'] > 1 else 'cold'} last 10 ({r['last10']:.1f} vs {r['season']:.1f})")
        return "; ".join(bits)

    def stocks(self, period: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
        """(teams, players): steals/blocks made and given up per game, ranked (1 = most)."""
        if self.model is None:
            raise RuntimeError("Load stats first (Stats tab → Update stats).")
        period = check_period(period)
        if "stl" in self.model.missing.get(period, set()):
            raise RuntimeError("No steals/blocks data for this period yet - click Update stats.")
        h = self.model.hist.get(period)
        if h is None:
            raise RuntimeError("No full-game stats yet - click Update stats.")
        own = h.groupby(["team", "game_id"])[["stl", "blk"]].sum().groupby(level="team").mean()
        allowed = h.groupby(["opp", "game_id"])[["stl", "blk"]].sum().groupby(level="opp").mean()
        allowed.index.name = "team"
        teams = own.join(allowed, rsuffix="_allowed")
        teams["games"] = h.groupby("team")["game_id"].nunique()
        for c in ["stl", "blk", "stl_allowed", "blk_allowed"]:
            teams[f"{c}_rank"] = teams[c].rank(ascending=False, method="min").astype(int)
        p = self.model.players[period]
        players = pd.DataFrame({
            "player": p["player_name"], "team": p["team"], "pos": p["pos"], "games": p["games"],
            "min": p["min_avg"], "stl": p["avg_stl"], "blk": p["avg_blk"],
            "sb": p["avg_stl"] + p["avg_blk"], "l10_stl": p["last10_stl"], "l10_blk": p["last10_blk"],
        })
        players = players[players["games"] >= config.MIN_GAMES]
        players["stl_rank"] = players["stl"].rank(ascending=False, method="min").astype(int)
        players["blk_rank"] = players["blk"].rank(ascending=False, method="min").astype(int)
        return teams, players.sort_values("sb", ascending=False)

    # ------------------------------------------------------------- simulations
    def simulate_games(self, day=None, n: int = config.SIM_DEFAULT_RUNS) -> tuple[pd.DataFrame, str]:
        if self.model is None:
            raise RuntimeError("Load stats first (Stats tab → Update stats).")
        from .sim import Simulator
        games, src = self.schedule(day)
        if games.empty:
            return pd.DataFrame(), f"No games scheduled ({src})."
        res = Simulator(self.model, self.context(day)).run_games(games, int(n))
        return res, f"{len(games)} games, {int(n):,} simulations each ({src})"

    def simulate_parlay(self, legs: list[dict], day=None, n: int = config.SIM_DEFAULT_RUNS,
                        odds=None) -> dict:
        if self.model is None:
            raise RuntimeError("Load stats first (Stats tab → Update stats).")
        if not legs:
            raise ValueError("Add at least one leg.")
        from .sim import simulate_parlay
        games, _ = self.schedule(day)
        return simulate_parlay(self.model, self.context(day), games, legs, int(n), odds)

    def parse_legs(self, text: str, day=None) -> dict:
        """Read parlay legs out of pasted text (see q1model/paste.py)."""
        if self.model is None:
            raise RuntimeError("Load stats first (Stats tab → Update stats).")
        from .paste import Parser
        d = self.model.directory
        return Parser(dict(zip(d["player_name"], d["team"])), set(self.tonight_teams(day))).parse(text)

    def tonight_teams(self, day=None) -> list[str]:
        try:
            games, _ = self.schedule(day)
        except RuntimeError:
            return []
        return sorted(set(games["away"]) | set(games["home"]))

    def game_labels(self, day=None) -> list[str]:
        games, _ = self.schedule(day)
        return [f"{g.away} @ {g.home}" for g in games.itertuples()]

    # ------------------------------------------------------------- rankings
    def rankings_all(self, period: str = "full") -> dict:
        period = check_period(period)
        h = self.model.hist.get(period)
        if h is None:
            raise RuntimeError("No full-game stats yet - click Update stats.")
        return defense.all_rankings(h)

    def rankings(self, stat: str, period: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.model is None:
            raise RuntimeError("Load stats first (Stats tab → Update stats).")
        period = check_period(period)
        h = self.model.hist.get(period)
        if h is None:
            raise RuntimeError("No full-game stats yet - click Update stats.")
        return defense.rankings(h, stat)

    # ------------------------------------------------------------- bets
    def bets(self) -> pd.DataFrame:
        return tracker.load(self.paths["bets"])

    def log_bet(self, **kw) -> int:
        kw.setdefault("game_date", date.today().isoformat())
        kw["period"] = check_period(kw.get("period", "q1"))
        kw["odds"] = _odds(kw["odds"])
        kw["stake"] = float(kw["stake"])
        kw["line"] = float(kw["line"])
        kw["opp"] = str(kw["opp"]).upper()
        return tracker.add_bet(self.paths["bets"], **kw)

    def set_close(self, bet_id: int, close_odds):
        tracker.set_close(self.paths["bets"], int(bet_id), _odds(close_odds))

    def delete_bet(self, bet_id: int):
        tracker.delete_bet(self.paths["bets"], int(bet_id))

    def settle(self) -> int:
        if not self.has_data():
            return 0
        return tracker.settle(self.paths["bets"], data.load_table(self.data_dir),
                              data.load_full_table(self.data_dir))

    def report(self) -> str:
        return tracker.report(self.paths["bets"])

    # ------------------------------------------------------------- testing
    def backtest(self, period: str = "q1", **kw) -> str:
        return backtest.run(data.load_table(self.data_dir), data.load_full_table(self.data_dir),
                            period, **kw)

    def tune(self, period: str = "q1", **kw) -> str:
        return backtest.tune(data.load_table(self.data_dir), data.load_full_table(self.data_dir),
                             period, **kw)


def _odds(v) -> float:
    s = str(v).strip().replace("+", "")
    o = float(s)
    if -100 < o < 100:
        raise ValueError(f"Odds should be American, like -110 or +120 (got {v})")
    return o


def resource_dir() -> Path:
    """Folder containing bundled files when frozen by PyInstaller."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
