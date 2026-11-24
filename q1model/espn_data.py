"""Stats from ESPN's public JSON feeds instead of stats.nba.com.

Used by the cloud (phone) version: stats.nba.com blocks most cloud servers, ESPN doesn't.
Produces exactly the same tables as data.py, so the model doesn't know the difference:
  * full-game rows come from each game's box score
  * 1st-quarter rows are added up from the play-by-play (minutes from substitutions),
    and a game's Q1 rows are only kept if they check out (team points match the
    official 1st-quarter score and each team's minutes add up to ~60).
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import data
from .data import COLUMNS
from .feeds import team_abbr

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={e}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; q1-model/1.0)"}
POS_LETTER = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C",
              "G-F": "G", "F-G": "F", "F-C": "F", "C-F": "C"}


def _get(url: str, tries: int = 3) -> dict:
    import requests
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    return {}


def season_label(d: date) -> str:
    y = d.year if d.month >= 8 else d.year - 1
    return f"{y}-{str(y + 1)[2:]}"


# ---------------------------------------------------------------- scoreboard
def finished_games(d: date) -> list[str]:
    """ESPN event ids of regular-season / Cup games that are final on date d."""
    js = _get(SCOREBOARD.format(d=d.strftime("%Y%m%d")))
    out = []
    for ev in js.get("events", []):
        st = ev.get("status", {}).get("type", {})
        season_type = ev.get("season", {}).get("type", 2)
        if st.get("completed") and season_type in (2, 6, "2", "6"):   # 2 regular, 6 = NBA Cup final
            out.append(str(ev["id"]))
    return out


# ---------------------------------------------------------------- helpers
def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _made_att(v) -> tuple[float, float]:
    s = str(v or "0-0")
    if "-" in s:
        a, b = s.split("-", 1)
        return _num(a), _num(b)
    return _num(s), 0.0


def _clock_secs(v) -> float:
    s = str(v or "0")
    if ":" in s:
        m, sec = s.split(":", 1)
        return _num(m) * 60 + _num(sec)
    return _num(s)


def _teams(js: dict) -> dict:
    """{espn team id: dict(abbr, home, q1_pts)} from the game header."""
    comp = js["header"]["competitions"][0]
    out = {}
    for c in comp["competitors"]:
        tid = str(c.get("id") or c["team"]["id"])
        ls = c.get("linescores") or []
        q1 = _num(ls[0].get("value", ls[0].get("displayValue"))) if ls else None
        out[tid] = dict(abbr=team_abbr(c["team"].get("abbreviation")) or c["team"].get("abbreviation"),
                        home=c.get("homeAway") == "home", q1_pts=q1)
    return out


# ---------------------------------------------------------------- full game
def parse_full(js: dict, gid: str, gdate: date) -> pd.DataFrame:
    teams = _teams(js)
    rows = []
    for tb in js.get("boxscore", {}).get("players", []):
        tid = str(tb["team"]["id"])
        me = teams.get(tid)
        if not me:
            continue
        opp = next(v["abbr"] for k, v in teams.items() if k != tid)
        st = tb["statistics"][0]
        labels = [str(x).upper() for x in (st.get("labels") or st.get("names") or [])]
        idx = {lab: i for i, lab in enumerate(labels)}
        for a in st.get("athletes", []):
            stats = a.get("stats") or []
            if a.get("didNotPlay") or not stats:
                continue

            def g(lab):
                i = idx.get(lab)
                return stats[i] if i is not None and i < len(stats) else "0"
            mins = _num(str(g("MIN")).split(":")[0])
            if mins <= 0:
                continue
            fg3m, _ = _made_att(g("3PT"))
            _, fga = _made_att(g("FG"))
            _, fta = _made_att(g("FT"))
            ath = a["athlete"]
            pos = (ath.get("position") or {}).get("abbreviation", "")
            rows.append(dict(
                game_id=gid, game_date=gdate.isoformat(), season=season_label(gdate),
                player_id=int(ath["id"]), player_name=ath.get("displayName", ""), team=me["abbr"], opp=opp,
                home=me["home"], starter=bool(a.get("starter")),
                pos=POS_LETTER.get(pos, "") if a.get("starter") else "",
                min=mins, pts=_num(g("PTS")), reb=_num(g("REB")), ast=_num(g("AST")), fg3m=fg3m,
                fga=fga, fta=fta, tov=_num(g("TO")), stl=_num(g("STL")), blk=_num(g("BLK"))))
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------------------------------------------------------------- 1st quarter
def parse_q1(js: dict, full: pd.DataFrame, gid: str, gdate: date) -> pd.DataFrame:
    """Add up the first quarter from the play-by-play. Empty frame if it doesn't check out."""
    teams = _teams(js)
    if full.empty:
        return pd.DataFrame(columns=COLUMNS)
    info = full.set_index("player_id")
    team_of = info["team"].to_dict()
    stat = {pid: dict(pts=0.0, reb=0.0, ast=0.0, fg3m=0.0, fga=0.0, fta=0.0, tov=0.0, stl=0.0, blk=0.0)
            for pid in info.index}
    mins = {pid: 0.0 for pid in info.index}
    on = {t: set(info.index[(info["team"] == t) & info["starter"]]) for t in set(team_of.values())}
    since = {pid: 720.0 for s in on.values() for pid in s}       # clock when he checked in
    touched = set()

    def bump(pid, k, v=1.0):
        if pid in stat:
            stat[pid][k] += v
            touched.add(pid)

    for p in js.get("plays", []):
        if (p.get("period") or {}).get("number") != 1:
            continue
        typ = str((p.get("type") or {}).get("text", "")).lower()
        txt = str(p.get("text", "")).lower()
        parts = [int(x["athlete"]["id"]) for x in p.get("participants", [])
                 if str((x.get("athlete") or {}).get("id", "")).isdigit()]
        clock = _clock_secs((p.get("clock") or {}).get("displayValue"))
        if "substitution" in typ or " enters the game for " in txt:
            if len(parts) >= 2:
                inn, out_ = parts[0], parts[1]
                # make sure the order matches the text: "<in> enters the game for <out>"
                name_in = str(info.loc[out_, "player_name"]).lower() if out_ in info.index else ""
                if name_in and txt.startswith(name_in):
                    inn, out_ = out_, inn
                t = team_of.get(inn) or team_of.get(out_)
                if out_ in since:
                    mins[out_] += (since.pop(out_) - clock) / 60
                if t is not None:
                    on.setdefault(t, set()).discard(out_)
                    on[t].add(inn)
                since[inn] = clock
                touched.update({inn, out_})
            continue
        if not parts:
            continue
        a = parts[0]
        if "free throw" in typ:
            bump(a, "fta")
            if p.get("scoringPlay"):
                bump(a, "pts", _num(p.get("scoreValue")) or 1)
            continue
        if p.get("shootingPlay"):
            bump(a, "fga")
            if p.get("scoringPlay"):
                val = _num(p.get("scoreValue"))
                bump(a, "pts", val)
                if val == 3:
                    bump(a, "fg3m")
                if len(parts) > 1 and "assist" in txt:
                    bump(parts[1], "ast")
            elif len(parts) > 1 and "block" in txt:
                bump(parts[1], "blk")
            continue
        if "rebound" in typ:
            bump(a, "reb")
            continue
        if "turnover" in typ or "bad pass" in typ or "lost ball" in typ or "traveling" in typ \
                or "turnover" in txt:
            bump(a, "tov")
            if len(parts) > 1 and "steal" in txt:
                bump(parts[1], "stl")
    for pid, t0 in since.items():           # still on the floor at the end of the quarter
        mins[pid] += t0 / 60

    rows = []
    for pid in touched | {p for p, m in mins.items() if m > 0}:
        if pid not in info.index or mins.get(pid, 0) <= 0:
            continue
        r = info.loc[pid]
        rows.append(dict(game_id=gid, game_date=gdate.isoformat(), season=season_label(gdate), player_id=pid,
                         player_name=r["player_name"], team=r["team"], opp=r["opp"], home=r["home"],
                         starter=bool(r["starter"]), pos=r["pos"], min=round(mins[pid], 3), **stat[pid]))
    q1 = pd.DataFrame(rows, columns=COLUMNS)
    # sanity checks: ~60 minutes per team and points match the official 1st-quarter score
    for tid, t in teams.items():
        sub = q1[q1["team"] == t["abbr"]]
        if abs(sub["min"].sum() - 60) > 3:
            return pd.DataFrame(columns=COLUMNS)
        if t["q1_pts"] is not None and abs(sub["pts"].sum() - t["q1_pts"]) > 2:
            return pd.DataFrame(columns=COLUMNS)
    return q1


# ---------------------------------------------------------------- update
def update(data_dir: str | Path, start: date | None = None, end: date | None = None,
           progress=print, max_days: int | None = None) -> str:
    """Download every finished game from `start` (default: season start / last saved day)
    through `end` (default: yesterday). Safe to run as often as you like."""
    p = data.paths(data_dir)
    p["box"].mkdir(parents=True, exist_ok=True)
    p["full"].mkdir(parents=True, exist_ok=True)
    end = end or (date.today() - timedelta(days=1))
    have = {f.stem for f in p["box"].glob("*.csv")} | {f.stem.replace("full_", "") for f in p["full"].glob("full_*.csv")}
    if start is None:
        y = end.year if end.month >= 8 else end.year - 1
        start = date(y, 10, 15)
    d, days, new_games, q1_bad = start, 0, 0, 0
    while d <= end:
        try:
            ids = finished_games(d)
        except Exception as e:  # noqa: BLE001
            progress(f"{d}: scoreboard failed ({e})")
            d += timedelta(days=1)
            continue
        for gid in ids:
            if gid in have:
                continue
            try:
                js = _get(SUMMARY.format(e=gid))
                full = parse_full(js, gid, d)
                if full.empty:
                    continue
                full.to_csv(p["full"] / f"full_{gid}.csv", index=False)
                q1 = parse_q1(js, full, gid, d)
                if q1.empty:
                    q1_bad += 1
                    pd.DataFrame(columns=COLUMNS).to_csv(p["box"] / f"{gid}.csv", index=False)
                else:
                    q1.to_csv(p["box"] / f"{gid}.csv", index=False)
                new_games += 1
                progress(f"{d}: saved game {gid} ({full['team'].iloc[0]} vs {full['opp'].iloc[0]})")
                time.sleep(0.3)
            except Exception as e:  # noqa: BLE001
                progress(f"{d}: game {gid} failed ({e})")
        d += timedelta(days=1)
        days += 1
        if max_days and days >= max_days:
            break
    if new_games or not p["table"].exists():
        rebuild(data_dir)
    return (f"Added {new_games} new games through {end:%b %d}"
            + (f" ({q1_bad} without usable 1st-quarter play-by-play)" if q1_bad else "") + ".")


def rebuild(data_dir: str | Path) -> None:
    p = data.paths(data_dir)
    qs = [pd.read_csv(f, dtype={"game_id": str, "pos": str}) for f in p["box"].glob("*.csv")]
    qs = [q for q in qs if len(q)]
    q1 = (pd.concat(qs, ignore_index=True) if qs else pd.DataFrame(columns=COLUMNS))
    q1 = q1.reindex(columns=COLUMNS).sort_values(["game_date", "game_id", "team", "player_id"])
    q1.to_csv(p["table"], index=False)
    fs = [pd.read_csv(f, dtype={"game_id": str, "pos": str}) for f in p["full"].glob("full_*.csv")]
    if fs:
        full = pd.concat(fs, ignore_index=True).reindex(columns=COLUMNS)
        full = full.sort_values(["game_date", "game_id", "team", "player_id"])
        full.to_csv(p["full_table"], index=False)
