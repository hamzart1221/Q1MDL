"""Phone report: one self-contained HTML page (no scripts needed) with tonight's
favorite picks, best spots, every game's players with full + 1st-quarter lookups,
and injuries. Save it to OneDrive / iCloud Drive / Google Drive and open it on your phone."""
from __future__ import annotations

import html
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .core import STAT_LABELS

PERIOD_NAME = {"full": "Full game", "q1": "1st quarter"}

CSS = """
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0b1020;color:#e8ecf5;font-size:15px}
header{background:linear-gradient(90deg,#1d428a,#c8102e);padding:14px 16px}
header h1{margin:0;font-size:20px}header p{margin:4px 0 0;opacity:.85;font-size:13px}
nav{position:sticky;top:0;background:#111832;display:flex;overflow-x:auto;gap:4px;padding:8px;z-index:5;border-bottom:1px solid #263056}
nav a{color:#e8ecf5;text-decoration:none;background:#1b2447;padding:7px 11px;border-radius:16px;white-space:nowrap;font-size:13px}
section{padding:10px 12px}h2{font-size:17px;margin:14px 0 8px;color:#fdb927}h3{font-size:15px;margin:12px 0 6px}
.card{background:#141c38;border:1px solid #263056;border-radius:12px;padding:10px 12px;margin:8px 0}
.row{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.pick{font-weight:700;font-size:16px}.over{color:#3ddc84}.under{color:#ff5a6e}
.muted{color:#9aa4c2;font-size:12.5px}.big{font-size:18px;font-weight:700}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11.5px;background:#263056;margin-left:4px}
details{background:#141c38;border:1px solid #263056;border-radius:10px;margin:6px 0}
summary{padding:10px 12px;cursor:pointer;list-style:none}summary::-webkit-details-marker{display:none}
summary:after{content:"›";float:right;color:#9aa4c2}details[open] summary:after{content:"⌄"}
pre{white-space:pre-wrap;font-family:Menlo,Consolas,monospace;font-size:11.5px;line-height:1.35;margin:0;padding:0 12px 12px;color:#dfe5f3}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:6px 4px;border-bottom:1px solid #263056;text-align:left}
th{color:#9aa4c2;font-weight:600}
.u{color:#3ddc84}.d{color:#ff5a6e}
"""


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _color(text: str) -> str:
    """Escape, then color ▲/▼ lines' arrows."""
    t = _e(text)
    return t.replace("▲", '<b class="u">▲</b>').replace("▼", '<b class="d">▼</b>')


def default_folder() -> Path:
    """A synced folder if one exists, so the report shows up on the phone by itself."""
    home = Path.home()
    cands = []
    if os.environ.get("OneDrive"):
        cands.append(Path(os.environ["OneDrive"]))
    if sys.platform == "darwin":
        cands.append(home / "Library" / "Mobile Documents" / "com~apple~CloudDocs")
    cands += [home / "OneDrive", home / "iCloudDrive", home / "Google Drive" / "My Drive", home / "Dropbox"]
    for c in cands:
        if c.exists():
            return c / "Q1 Model phone"
    return home / "Documents" / "Q1 Model" / "phone"


def build(core, day=None, progress=print) -> str:
    day = pd.Timestamp(day or pd.Timestamp.now().date()).date()
    parts = []
    nav = [("favs", "★ Picks"), ("spots", "Best spots"), ("games", "Games & lookups"), ("inj", "Injuries")]

    # ---- favorite picks
    progress("Phone report: favorite picks...")
    out = ['<section id="favs"><h2>★ Favorite Picks of the Day</h2>']
    try:
        favs, msg = core.favorite_picks(day)
        out.append(f'<p class="muted">{_e(msg)}. Over = highest line he clears ~60% of the time; '
                   'Under = lowest line he stays under ~60%. A better book line is even better.</p>')
        for side, title in (("over", "Overs"), ("under", "Unders")):
            sub = favs[favs["side"] == side] if len(favs) else favs
            if not len(sub):
                continue
            out.append(f"<h3>{title}</h3>")
            for r in sub.itertuples():
                out.append(
                    f'<div class="card"><div class="row"><span class="pick {side}">{side[0].upper()}{r.rank}. '
                    f'{_e(r.player)}</span><span class="big">{r.chance:.0%}</span></div>'
                    f'<div class="row"><span class="pick {side}">{_e(r.pick)}</span>'
                    f'<span class="muted">proj {r.proj:.1f} · avg {r.season:.1f}</span></div>'
                    f'<div class="muted">{_e(r.team)} vs {_e(r.opp)} · {_e(r.why)}</div></div>')
    except Exception as e:  # noqa: BLE001
        out.append(f'<p class="muted">Favorite picks unavailable: {_e(e)}</p>')
    out.append("</section>")
    parts.append("".join(out))

    # ---- best spots
    progress("Phone report: best spots...")
    out = ['<section id="spots"><h2>Best spots</h2>']
    try:
        bs, msg = core.best_spots_all(day)
        for period in ("full", "q1"):
            sub = bs[bs["period"] == period].head(25) if len(bs) else bs
            if not len(sub):
                continue
            out.append(f"<h3>{PERIOD_NAME[period]} - top {len(sub)}</h3><table>"
                       "<tr><th>Rating</th><th>Player</th><th>Stat</th><th>Proj</th><th>Avg</th></tr>")
            for r in sub.itertuples():
                cls = "u" if r.rating >= 1 else "d"
                out.append(f'<tr><td class="{cls}"><b>{r.rating:.2f}</b></td><td>{_e(r.player)}'
                           f'<div class="muted">{_e(r.team)} vs {_e(r.opp)} · {_e(r.category)}</div></td>'
                           f'<td>{_e(STAT_LABELS.get(r.stat, r.stat))}</td><td>{r.proj:.1f}</td>'
                           f'<td>{r.season:.1f}</td></tr>')
            out.append("</table>")
    except Exception as e:  # noqa: BLE001
        out.append(f'<p class="muted">Best spots unavailable: {_e(e)}</p>')
    out.append("</section>")
    parts.append("".join(out))

    # ---- every game, every rotation player, with lookups
    out = ['<section id="games"><h2>Games &amp; player lookups</h2>'
           '<p class="muted">Tap a player for the model\'s full write-up: projections, commentary and his '
           'last 10 games (▲ better than expected, ▼ worse, defense #1 = toughest).</p>']
    try:
        sl, msg = core.slate(day, "full")
        sl = sl[sl["minutes"] >= 15] if len(sl) else sl
        n = len(sl)
        for gi, (game, grp) in enumerate(sl.groupby("game", sort=False)):
            tip = grp["tip"].iloc[0] if "tip" in grp else ""
            out.append(f'<h3>{_e(game)} <span class="muted">{_e(tip)}</span></h3>')
            for r in grp.sort_values(["team", "minutes"], ascending=[True, False]).itertuples():
                progress(f"Phone report: lookups {_e(game)} ({gi + 1}) - {r.player}")
                status = f'<span class="tag">{_e(r.status)}</span>' if str(r.status) not in ("", "nan") else ""
                head = (f'<summary><b>{_e(r.player)}</b> <span class="muted">{_e(r.team)} · {r.minutes:.0f} min · '
                        f'{r.pts:.1f} pts {r.reb:.1f} reb {r.ast:.1f} ast</span>{status}</summary>')
                body = []
                for period in ("full", "q1"):
                    if period not in core.model.players:
                        continue
                    try:
                        txt = core.lookup(r.player, r.opp, None, period=period, day=day)
                    except Exception as e:  # noqa: BLE001
                        txt = f"Lookup unavailable: {e}"
                    body.append(f"<details><summary>{PERIOD_NAME[period]}</summary><pre>{_color(txt)}</pre>"
                                "</details>")
                out.append(f"<details>{head}<div style='padding:0 8px 8px'>{''.join(body)}</div></details>")
        if not n:
            out.append(f'<p class="muted">{_e(msg)}</p>')
    except Exception as e:  # noqa: BLE001
        out.append(f'<p class="muted">Slate unavailable: {_e(e)}</p>')
    out.append("</section>")
    parts.append("".join(out))

    # ---- injuries
    out = ['<section id="inj"><h2>Injuries</h2>']
    try:
        inj = core.injuries.combined()
        if len(inj):
            out.append("<table><tr><th>Player</th><th>Team</th><th>Status</th></tr>")
            for r in inj.itertuples():
                out.append(f"<tr><td>{_e(r.player)}</td><td>{_e(r.team)}</td><td>{_e(r.status)}</td></tr>")
            out.append("</table>")
        else:
            out.append('<p class="muted">No injuries listed.</p>')
    except Exception as e:  # noqa: BLE001
        out.append(f'<p class="muted">Injuries unavailable: {_e(e)}</p>')
    out.append("</section>")
    parts.append("".join(out))

    made = datetime.now().strftime("%b %d, %I:%M %p")
    navhtml = "".join(f'<a href="#{i}">{_e(t)}</a>' for i, t in nav)
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Q1 Model - {day:%b %d}</title><style>{CSS}</style></head><body>"
            f"<header><h1>🏀 Q1 Model - {day:%A, %b %d}</h1><p>Made {made} · re-make it after injury "
            "news to refresh</p></header>"
            f"<nav>{navhtml}</nav>{''.join(parts)}"
            "<p class='muted' style='padding:12px'>For entertainment and research. Bet responsibly.</p>"
            "</body></html>")


def save(core, day=None, folder: Path | None = None, progress=print) -> Path:
    folder = Path(folder) if folder else default_folder()
    folder.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp(day or pd.Timestamp.now().date()).date()
    page = build(core, day, progress)
    path = folder / f"Q1 Model {day:%Y-%m-%d}.html"
    path.write_text(page, encoding="utf-8")
    (folder / "Q1 Model - latest.html").write_text(page, encoding="utf-8")
    return path
