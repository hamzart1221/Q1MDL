"""Download and cache first-quarter player box scores from stats.nba.com (via nba_api).

Canonical table (one row per player per game, Q1 only):
    game_id, game_date, season, player_id, player_name, team, opp, home,
    starter, pos, min, pts, reb, ast, fg3m, fga, fta, tov
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

from . import config

COLUMNS = ["game_id", "game_date", "season", "player_id", "player_name", "team",
           "opp", "home", "starter", "pos", "min", "pts", "reb", "ast", "fg3m", "fga",
           "fta", "tov", "stl", "blk"]
NEW_STATS = ["stl", "blk"]  # added later; older cached files lack them


# --------------------------------------------------------------------------- paths
def paths(data_dir: str | Path) -> dict[str, Path]:
    d = Path(data_dir)
    return {
        "root": d,
        "box": d / "q1_box",
        "games": d / "games",
        "table": d / "q1_player_games.csv",
        "full": d / "full_games",
        "full_table": d / "full_player_games.csv",
        "bets": d / "bets.csv",
    }


def load_table(data_dir: str | Path) -> pd.DataFrame:
    p = paths(data_dir)["table"]
    if not p.exists():
        raise SystemExit(f"No data at {p}. Run `python q1.py update` first "
                         f"(or `python q1.py demo` to try it on fake data).")
    df = pd.read_csv(p, dtype={"game_id": str, "pos": str})
    df["pos"] = df["pos"].fillna("")
    for c in NEW_STATS:
        if c not in df:
            df[c] = float("nan")
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_full_table(data_dir: str | Path) -> pd.DataFrame | None:
    """Full-game player rows (same columns as the Q1 table), or None if not downloaded."""
    p = paths(data_dir)["full_table"]
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={"game_id": str, "pos": str})
    df["pos"] = df["pos"].fillna("")
    for c in NEW_STATS:
        if c not in df:
            df[c] = float("nan")
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def rebuild_full_table(data_dir: str | Path, q1: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Combine cached full-game logs; starter/position come from the Q1 box scores."""
    p = paths(data_dir)
    files = sorted(p["full"].glob("*.csv")) if p["full"].exists() else []
    if not files:
        return None
    df = pd.concat((pd.read_csv(f, dtype={"game_id": str}) for f in files), ignore_index=True)
    if q1 is None and p["table"].exists():
        q1 = pd.read_csv(p["table"], dtype={"game_id": str, "pos": str})
    if q1 is not None:
        sp = q1[["game_id", "player_id", "starter", "pos"]].drop_duplicates(["game_id", "player_id"])
        df = df.drop(columns=["starter", "pos"], errors="ignore").merge(
            sp, on=["game_id", "player_id"], how="left")
    df["starter"] = df.get("starter", False)
    df["starter"] = df["starter"].fillna(False).astype(bool)
    df["pos"] = df.get("pos", "")
    df["pos"] = df["pos"].fillna("")
    df = df.reindex(columns=COLUMNS).sort_values(["game_date", "game_id", "team", "player_id"])
    df.to_csv(p["full_table"], index=False)
    return df


def rebuild_table(data_dir: str | Path) -> pd.DataFrame:
    """Concatenate every cached per-game file into the single modelling table."""
    p = paths(data_dir)
    files = sorted(p["box"].glob("*.csv"))
    if not files:
        raise SystemExit("No cached box scores found.")
    df = pd.concat((pd.read_csv(f, dtype={"game_id": str}) for f in files),
                   ignore_index=True)
    df = df.reindex(columns=COLUMNS).sort_values(["game_date", "game_id", "team", "player_id"])
    df.to_csv(p["table"], index=False)
    rebuild_full_table(data_dir, df)
    return df


# --------------------------------------------------------------------------- helpers
def parse_minutes(v) -> float:
    """Handle '9:34', '9:34.000', 'PT09M34.00S', 9.5, None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return 0.0
    m = re.match(r"PT(\d+)M([\d.]+)S", s)
    if m:
        return int(m.group(1)) + float(m.group(2)) / 60
    if ":" in s:
        mm, ss = s.split(":", 1)
        return int(float(mm)) + float(ss) / 60
    try:
        return float(s)
    except ValueError:
        return 0.0


def _call(fn, **kwargs):
    last = None
    for attempt in range(config.REQUEST_RETRIES):
        try:
            time.sleep(config.REQUEST_PAUSE_SEC)
            return fn(timeout=config.REQUEST_TIMEOUT_SEC, **kwargs)
        except Exception as e:  # network hiccups, throttling
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Request failed after retries: {last}")


# --------------------------------------------------------------------------- schedule
def fetch_games(season: str, data_dir: str | Path, refresh: bool = True) -> pd.DataFrame:
    """One row per (game, team) for completed regular-season + playoff games."""
    from nba_api.stats.endpoints import leaguegamelog

    p = paths(data_dir)
    p["games"].mkdir(parents=True, exist_ok=True)
    out = p["games"] / f"{season}.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, dtype={"game_id": str})

    frames = []
    for st in ["Regular Season", "Playoffs"]:
        res = _call(leaguegamelog.LeagueGameLog, season=season,
                    season_type_all_star=st, player_or_team_abbreviation="T")
        g = res.get_data_frames()[0]
        if not g.empty:
            frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["game_id", "game_date", "team", "opp", "home"])
    g = pd.concat(frames, ignore_index=True)
    games = pd.DataFrame({
        "game_id": g["GAME_ID"].astype(str),
        "game_date": pd.to_datetime(g["GAME_DATE"]).dt.strftime("%Y-%m-%d"),
        "team": g["TEAM_ABBREVIATION"],
        "opp": g["MATCHUP"].str.split().str[-1],
        "home": g["MATCHUP"].str.contains("vs."),
    })
    games["season"] = season
    games.to_csv(out, index=False)
    return games


# --------------------------------------------------------------------------- full games
def fetch_full_logs(season: str, data_dir: str | Path) -> pd.DataFrame:
    """Every player's full-game box score for a season, in one request per season type."""
    from nba_api.stats.endpoints import leaguegamelog

    p = paths(data_dir)
    p["full"].mkdir(parents=True, exist_ok=True)
    frames = []
    for st in ["Regular Season", "Playoffs"]:
        res = _call(leaguegamelog.LeagueGameLog, season=season,
                    season_type_all_star=st, player_or_team_abbreviation="P")
        g = res.get_data_frames()[0]
        if not g.empty:
            frames.append(g)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    g = pd.concat(frames, ignore_index=True)
    df = pd.DataFrame({
        "game_id": g["GAME_ID"].astype(str),
        "game_date": pd.to_datetime(g["GAME_DATE"]).dt.strftime("%Y-%m-%d"),
        "season": season,
        "player_id": g["PLAYER_ID"].astype(int),
        "player_name": g["PLAYER_NAME"],
        "team": g["TEAM_ABBREVIATION"],
        "opp": g["MATCHUP"].str.split().str[-1],
        "home": g["MATCHUP"].str.contains("vs."),
        "starter": False, "pos": "",
        "min": g["MIN"].map(parse_minutes),
        "pts": g["PTS"], "reb": g["REB"], "ast": g["AST"], "fg3m": g["FG3M"],
        "fga": g["FGA"], "fta": g["FTA"], "tov": g["TOV"],
        "stl": g["STL"], "blk": g["BLK"],
    })
    df = df[df["min"] > 0]
    df.to_csv(p["full"] / f"{season}.csv", index=False)
    return df


# --------------------------------------------------------------------------- Q1 box
def _normalize_v3(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": df["personId"].astype(int),
        "player_name": (df["firstName"].fillna("") + " " + df["familyName"].fillna("")).str.strip(),
        "team": df["teamTricode"],
        "starter": df["position"].fillna("").astype(str).str.len() > 0,
        "pos": df["position"].fillna("").astype(str).str[:1],
        "min": df["minutes"].map(parse_minutes),
        "pts": df["points"], "reb": df["reboundsTotal"], "ast": df["assists"],
        "fg3m": df["threePointersMade"], "fga": df["fieldGoalsAttempted"],
        "fta": df["freeThrowsAttempted"], "tov": df["turnovers"],
        "stl": df["steals"], "blk": df["blocks"],
    })


def _normalize_v2(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": df["PLAYER_ID"].astype(int),
        "player_name": df["PLAYER_NAME"],
        "team": df["TEAM_ABBREVIATION"],
        "starter": df["START_POSITION"].fillna("").astype(str).str.len() > 0,
        "pos": df["START_POSITION"].fillna("").astype(str).str[:1],
        "min": df["MIN"].map(parse_minutes),
        "pts": df["PTS"], "reb": df["REB"], "ast": df["AST"], "fg3m": df["FG3M"],
        "fga": df["FGA"], "fta": df["FTA"], "tov": df["TO"],
        "stl": df["STL"], "blk": df["BLK"],
    })


def _looks_like_q1(box: pd.DataFrame) -> bool:
    """A real Q1 box has ~60 player-minutes per team (5 on court x 12 min)."""
    if box.empty:
        return False
    team_min = box.groupby("team")["min"].sum()
    return bool(((team_min > 45) & (team_min < 75)).all())


def fetch_q1_box(game_id: str) -> pd.DataFrame:
    """Return Q1-only player stats for one game.

    stats.nba.com filters a box score by period through range parameters whose
    behaviour has shifted over time, so several parameter sets are tried and the
    first one whose minutes add up to a single quarter is used.
    """
    from nba_api.stats.endpoints import boxscoretraditionalv2, boxscoretraditionalv3

    attempts = [
        (boxscoretraditionalv3.BoxScoreTraditionalV3, _normalize_v3,
         dict(start_period=1, end_period=1, range_type=0, start_range=0, end_range=0)),
        (boxscoretraditionalv3.BoxScoreTraditionalV3, _normalize_v3,
         dict(start_period=1, end_period=1, range_type=2, start_range=0, end_range=7200)),
        (boxscoretraditionalv2.BoxScoreTraditionalV2, _normalize_v2,
         dict(start_period=1, end_period=1, range_type=2, start_range=0, end_range=7200)),
        (boxscoretraditionalv2.BoxScoreTraditionalV2, _normalize_v2,
         dict(start_period=1, end_period=1, range_type=0, start_range=0, end_range=0)),
    ]
    errors = []
    for endpoint, norm, kw in attempts:
        try:
            res = _call(endpoint, game_id=game_id, **kw)
            box = norm(res.get_data_frames()[0])
        except Exception as e:
            errors.append(f"{endpoint.__name__}{kw}: {e}")
            continue
        if _looks_like_q1(box):
            return box
        errors.append(f"{endpoint.__name__}{kw}: minutes don't look like one quarter")
    raise RuntimeError(f"Could not get a Q1 box for {game_id}:\n  " + "\n  ".join(errors))


def update(seasons: list[str], data_dir: str | Path, max_games: int | None = None,
           progress=None) -> pd.DataFrame:
    """Download any completed games not yet cached, then rebuild the table.

    progress(done, total, message) is called as games download (used by the app).
    """
    p = paths(data_dir)
    p["box"].mkdir(parents=True, exist_ok=True)
    fetched = failed = 0
    for season in seasons:
        if progress:
            progress(0, 1, f"{season}: downloading full-game stats...")
        try:
            full = fetch_full_logs(season, data_dir)
            print(f"{season}: {len(full)} full-game player rows")
        except Exception as e:  # keep going with Q1 data if this fails
            print(f"  ! full-game stats failed for {season}: {e}")
        games = fetch_games(season, data_dir)
        cached = {f.stem for f in p["box"].glob("*.csv")}
        # one-time: re-download Q1 box scores saved before steals/blocks were tracked
        stale = {f.stem for f in p["box"].glob("*.csv")
                 if not set(NEW_STATS) <= set(pd.read_csv(f, nrows=0).columns)}
        if stale:
            print(f"{season}: re-downloading {len(stale)} saved Q1 box scores to add steals/blocks")
        todo = sorted((set(games["game_id"]) - cached) | (stale & set(games["game_id"])))
        if max_games:
            todo = todo[:max_games]
        print(f"{season}: {games['game_id'].nunique()} games, {len(todo)} to download")
        if progress:
            progress(0, len(todo), f"{season}: {len(todo)} new games to download")
        for i, gid in enumerate(todo, 1):
            try:
                box = fetch_q1_box(gid)
            except RuntimeError as e:
                failed += 1
                print(f"  ! {e}")
                continue
            meta = games[games["game_id"] == gid].set_index("team")
            box = box[box["team"].isin(meta.index)].copy()
            box["game_id"] = gid
            box["game_date"] = meta["game_date"].iloc[0]
            box["season"] = season
            box["opp"] = box["team"].map(meta["opp"])
            box["home"] = box["team"].map(meta["home"])
            box = box[box["min"] > 0]           # DNPs / didn't play in Q1
            box[COLUMNS].to_csv(p["box"] / f"{gid}.csv", index=False)
            fetched += 1
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")
            if progress:
                progress(i, len(todo), f"{season}: {i}/{len(todo)} games")
    print(f"Downloaded {fetched} games, {failed} failed.")
    return rebuild_table(data_dir)
