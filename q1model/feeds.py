"""Tonight's schedule and the injury report.

Both come from ESPN's public JSON feeds (the same data espn.com shows). If ESPN is
unreachable, the schedule falls back to stats.nba.com via nba_api, and injuries fall
back to whatever was downloaded last plus your manual entries.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
HEADERS = {"User-Agent": "Mozilla/5.0 (Q1 Model desktop app)"}

TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}
_ALIASES = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS",
            "PHO": "PHX", "BK": "BKN", "BRK": "BKN", "CHO": "CHA", "NOR": "NOP",
            "LOS ANGELES CLIPPERS": "LAC", "LA LAKERS": "LAL"}
_BY_NAME = {v.upper(): k for k, v in TEAM_NAMES.items()}
_BY_NICK = {v.split()[-1].upper(): k for k, v in TEAM_NAMES.items()}  # "Celtics" -> BOS
_BY_NICK["BLAZERS"] = "POR"


def team_abbr(x: str) -> str | None:
    s = str(x or "").strip().upper()
    if s in TEAM_NAMES:
        return s
    if s in _ALIASES:
        return _ALIASES[s]
    if s in _BY_NAME:
        return _BY_NAME[s]
    last = s.split()[-1] if s else ""
    return _BY_NICK.get(last)


def _get_json(url: str, timeout: int = 15) -> dict:
    import requests
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- schedule
def parse_espn_scoreboard(js: dict) -> pd.DataFrame:
    rows = []
    for ev in js.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        teams = {}
        for c in comp.get("competitors", []):
            t = c.get("team", {})
            ab = team_abbr(t.get("abbreviation")) or team_abbr(t.get("displayName"))
            teams[c.get("homeAway")] = ab
        if teams.get("home") and teams.get("away"):
            tip = ev.get("date", "")
            try:
                tip = (pd.Timestamp(tip).tz_convert("America/New_York").strftime("%I:%M %p ET")
                       .lstrip("0"))
            except Exception:  # noqa: BLE001
                pass
            status = (comp.get("status") or ev.get("status") or {}).get("type", {}).get("shortDetail", "")
            rows.append({"away": teams["away"], "home": teams["home"], "tip": tip, "status": status})
    return pd.DataFrame(rows, columns=["away", "home", "tip", "status"])


def fetch_schedule(day: date | str) -> tuple[pd.DataFrame, str]:
    """Games on `day` as rows (away, home, tip, status). Returns (games, source)."""
    d = pd.Timestamp(day)
    errors = []
    try:
        games = parse_espn_scoreboard(_get_json(ESPN_SCOREBOARD.format(d=d.strftime("%Y%m%d"))))
        return games, "ESPN"
    except Exception as e:  # noqa: BLE001
        errors.append(f"ESPN: {e}")
    try:
        from nba_api.stats.endpoints import scoreboardv2
        from nba_api.stats.static import teams as static_teams
        ids = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
        gh = scoreboardv2.ScoreboardV2(game_date=d.strftime("%Y-%m-%d"), timeout=30).get_data_frames()[0]
        games = pd.DataFrame({"away": gh["VISITOR_TEAM_ID"].map(ids), "home": gh["HOME_TEAM_ID"].map(ids),
                              "tip": gh.get("GAME_STATUS_TEXT", ""), "status": ""})
        return games.drop_duplicates(), "NBA.com"
    except Exception as e:  # noqa: BLE001
        errors.append(f"NBA.com: {e}")
    raise RuntimeError("Couldn't download the schedule.\n" + "\n".join(errors))


# --------------------------------------------------------------------------- injuries
STATUS_MAP = {"OUT": "OUT", "DOUBTFUL": "DOUBTFUL", "QUESTIONABLE": "QUESTIONABLE",
              "DAY-TO-DAY": "QUESTIONABLE", "DAY TO DAY": "QUESTIONABLE", "GTD": "QUESTIONABLE",
              "GAME TIME DECISION": "QUESTIONABLE", "PROBABLE": "PROBABLE",
              "SUSPENSION": "SUSPENDED", "SUSPENDED": "SUSPENDED", "ACTIVE": "ACTIVE",
              "AVAILABLE": "ACTIVE", "INJURED RESERVE": "OUT", "OUT FOR SEASON": "OUT"}


def norm_status(s: str) -> str:
    s = str(s or "").strip().upper()
    return STATUS_MAP.get(s, "OUT" if s.startswith("OUT") else s or "QUESTIONABLE")


def parse_espn_injuries(js: dict) -> pd.DataFrame:
    rows = []
    for team_block in js.get("injuries", []):
        team = team_abbr(team_block.get("displayName")) or team_abbr(team_block.get("abbreviation"))
        for it in team_block.get("injuries", []):
            ath = it.get("athlete", {}) or {}
            name = ath.get("displayName") or ath.get("fullName")
            if not name:
                continue
            t = team or team_abbr((ath.get("team") or {}).get("abbreviation"))
            status = it.get("status") or (it.get("type") or {}).get("description", "")
            fantasy = ((it.get("details") or {}).get("fantasyStatus") or {}).get("description")
            note = it.get("shortComment") or it.get("longComment") or ""
            rows.append({"player": name, "team": t, "status": norm_status(status),
                         "detail": fantasy or status, "note": note[:200],
                         "updated": str(it.get("date", ""))[:10], "source": "ESPN"})
    return pd.DataFrame(rows, columns=["player", "team", "status", "detail", "note", "updated", "source"])


class Injuries:
    """Automatic (ESPN) + manual injury list. Manual entries always win."""

    COLS = ["player", "team", "status", "detail", "note", "updated", "source"]

    def __init__(self, home: Path):
        self.auto_path = Path(home) / "injuries_auto.csv"
        self.manual_path = Path(home) / "injuries_manual.csv"
        self.meta_path = Path(home) / "injuries_meta.json"

    def _read(self, p: Path) -> pd.DataFrame:
        if p.exists():
            try:
                return pd.read_csv(p).reindex(columns=self.COLS)
            except Exception:  # noqa: BLE001
                pass
        return pd.DataFrame(columns=self.COLS)

    @property
    def auto(self) -> pd.DataFrame:
        return self._read(self.auto_path)

    @property
    def manual(self) -> pd.DataFrame:
        return self._read(self.manual_path)

    def last_refresh(self) -> str:
        try:
            return json.loads(self.meta_path.read_text()).get("refreshed", "never")
        except Exception:  # noqa: BLE001
            return "never"

    def refresh(self) -> int:
        df = parse_espn_injuries(_get_json(ESPN_INJURIES))
        df.to_csv(self.auto_path, index=False)
        self.meta_path.write_text(json.dumps({"refreshed": datetime.now().strftime("%b %d %I:%M %p")}))
        return len(df)

    def set_manual(self, player: str, team: str, status: str, note: str = ""):
        m = self.manual
        from .model import norm_name
        m = m[m["player"].map(norm_name) != norm_name(player)]
        row = {"player": player.strip(), "team": team_abbr(team) or str(team).upper(),
               "status": norm_status(status), "detail": "manual", "note": note,
               "updated": date.today().isoformat(), "source": "Manual"}
        pd.concat([m, pd.DataFrame([row])], ignore_index=True).to_csv(self.manual_path, index=False)

    def remove_manual(self, player: str):
        from .model import norm_name
        m = self.manual
        m[m["player"].map(norm_name) != norm_name(player)].to_csv(self.manual_path, index=False)

    def combined(self) -> pd.DataFrame:
        from .model import norm_name
        a, m = self.auto, self.manual
        if len(m):
            a = a[~a["player"].map(norm_name).isin(set(m["player"].map(norm_name)))]
        return pd.concat([m, a], ignore_index=True)
