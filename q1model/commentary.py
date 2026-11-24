"""Plain-English write-up of what the model thinks about one player tonight."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import PERIOD_LABELS, prob_over_under

LABEL = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes",
         "stl": "steals", "blk": "blocks", "pra": "PRA", "pr": "pts+reb", "pa": "pts+ast",
         "ra": "reb+ast", "sb": "steals+blocks"}
POS_NAME = {"PG": "point guards", "SG": "shooting guards", "SF": "small forwards",
            "PF": "power forwards", "C": "centers"}
# a stat needs this season average (full game) before its "favorability" is worth mentioning
MIN_VOLUME = {"pts": 6, "reb": 2.5, "ast": 1.5, "fg3m": 1.2, "stl": 0.7, "blk": 0.6}
BREAKEVEN = 0.524  # win rate needed at -110


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _pct(x: float) -> str:
    return f"{x - 1:+.0%}"


def write(proj: dict[str, dict], ranks: dict[str, int | None], last: pd.DataFrame,
          period: str, line: float | None = None, line_stat: str = "pts") -> str:
    """proj: stat -> projection dict from Model.project (same player/opponent/context)."""
    any_r = next(iter(proj.values()))
    name, opp, pos = any_r["player_name"].split()[-1], any_r["opp"], any_r["pos"]
    full_name = any_r["player_name"]
    scale = 0.25 if period == "q1" else 1.0
    out = []

    # ---------------------------------------------------------------- headline
    if any_r["status"] in ("OUT", "DOUBTFUL", "SUSPENDED", "INACTIVE"):
        return (f"{full_name} is listed {any_r['status']} tonight, so the model projects nothing for him. "
                "If you know he's playing, set him to ACTIVE on the Injuries tab.")
    main = [s for s in ("pts", "reb", "ast") if s in proj]
    parts = [f"{proj[s]['mean']:.1f} {LABEL[s]}" for s in main]
    pra = proj.get("pra")
    head = f"The model has {full_name} at " + ", ".join(parts)
    if pra:
        ratio = pra["mean"] / pra["season_avg"] if pra["season_avg"] > 0 else 1
        if abs(ratio - 1) < 0.03:
            head += f" ({pra['mean']:.1f} PRA), right in line with his season average."
        else:
            head += (f" ({pra['mean']:.1f} PRA), about {abs(ratio - 1):.0%} "
                     f"{'above' if ratio > 1 else 'below'} his season average of {pra['season_avg']:.1f}.")
    else:
        head += "."
    if period == "q1":
        head = head.replace("The model has", "For the first quarter, the model has")
    out.append(head)

    # ---------------------------------------------------------------- drivers
    r = proj.get("pts") or any_r
    drivers = []
    mf = r["minutes_factor"]
    if r["minutes_boost"] > 0.3 * scale * 4 and r["teammates_out"]:
        usage = (f", and his share of the offense goes up {_pct(r['injury_boost'])}"
                 if r["injury_boost"] >= 1.01 else "")
        drivers.append(f"More minutes: with {', '.join(r['teammates_out'][:3])} out, he's projected for "
                       f"{r['minutes']:.1f} minutes ({r['minutes_boost']:+.1f} vs normal){usage}.")
    elif r["teammates_out"] and r["injury_boost"] > 1.01:
        drivers.append(f"Teammates out ({', '.join(r['teammates_out'][:3])}) bump his usage "
                       f"{_pct(r['injury_boost'])}.")
    elif abs(mf - 1) >= 0.05:
        drivers.append(f"Role: his minutes have {'risen' if mf > 1 else 'dropped'} lately "
                       f"({r['min_last10']:.1f} over the last 10 vs {r['min_season']:.1f} on the season).")

    l10, sea = r["last10_avg"], r["season_avg"]
    if sea > 0 and abs(l10 / sea - 1) >= 0.12:
        hot = l10 > sea
        drivers.append(f"Form: he's {'hot' if hot else 'cold'}, averaging {l10:.1f} points over his last "
                       f"10 vs {sea:.1f} on the season. The model leans {'up' if hot else 'down'} but "
                       f"doesn't chase it fully ({_pct(r['form_factor'])} on his scoring rate).")

    mfac, rk = r["matchup_factor"], ranks.get("pts")
    if rk:
        tone = ("one of the softest defenses" if rk <= 5 else "a soft defense" if rk <= 10 else
                "one of the toughest defenses" if rk >= 26 else "a tough defense" if rk >= 21 else
                "an average defense")
        drivers.append(f"Matchup: {opp} is {tone} against {POS_NAME.get(pos, pos)}, "
                       f"{ordinal(rk)} in points allowed to them ({mfac:.2f}x).")
    elif abs(mfac - 1) >= 0.04:
        drivers.append(f"Matchup: {opp} {'gives up more' if mfac > 1 else 'gives up less'} than average "
                       f"to {POS_NAME.get(pos, pos)} ({mfac:.2f}x).")

    if r["b2b"] and r["opp_b2b"]:
        drivers.append("Rest: both teams are on the second night of a back-to-back, roughly a wash.")
    elif r["b2b"]:
        drivers.append(f"Rest: {r['team']} is on the second night of a back-to-back ({_pct(r['b2b_factor'])}).")
    elif r["opp_b2b"]:
        drivers.append(f"Rest: {opp} is on the second night of a back-to-back, which helps him "
                       f"({_pct(r['b2b_factor'])}).")

    if r["vs_games"]:
        hf = r["h2h_factor"]
        extra = (f"; the model gives that {_pct(hf)}" if abs(hf - 1) >= 0.02 else
                 ", too small a sample to move the projection much")
        drivers.append(f"History: {r['vs_avg']:.1f} points in {r['vs_games']} game"
                       f"{'s' if r['vs_games'] > 1 else ''} vs {opp} this season{extra}.")
    if drivers:
        out.append("")
        out.append("What's driving it:")
        out += [f"  • {d}" for d in drivers]

    # ---------------------------------------------------------------- best / worst stats
    scored = []
    for s, p in proj.items():
        if s not in MIN_VOLUME or p["season_avg"] < MIN_VOLUME[s] * scale:
            continue
        scored.append((p["mean"] / p["season_avg"], s, ranks.get(s)))
    if len(scored) >= 2:
        scored.sort(reverse=True)
        best, worst = scored[0], scored[-1]
        out.append("")
        line_b = f"Best spot: {LABEL[best[1]]} ({best[0]:.2f}x his average"
        line_b += f", {opp} ranks {ordinal(best[2])} in {LABEL[best[1]]} allowed to {pos}s)." if best[2] else ")."
        out.append(line_b)
        if worst[0] < 0.98:
            line_w = f"Toughest: {LABEL[worst[1]]} ({worst[0]:.2f}x"
            line_w += f", {opp} ranks {ordinal(worst[2])})." if worst[2] else ")."
            out.append(line_w)

    # ---------------------------------------------------------------- line verdict
    if line is not None and line_stat in proj:
        p = proj[line_stat]
        po, pu, pp = prob_over_under(p["mean"], p["alpha"], float(line))
        side, prob = ("over", po) if po >= pu else ("under", pu)
        if prob >= 0.60:
            verdict = f"a strong lean to the {side}"
        elif prob >= 0.55:
            verdict = f"a lean to the {side}"
        elif prob >= BREAKEVEN:
            verdict = f"a slight lean to the {side}, but not enough to beat standard -110 juice"
        else:
            verdict = "basically a coin flip, no edge"
        hit = ""
        if len(last) and line_stat in last:
            n_over = int((last[line_stat] > float(line)).sum())
            hit = f" He's gone over {line:g} in {n_over} of his last {len(last)}."
        out.append("")
        out.append(f"On {line:g} {LABEL[line_stat]}: {prob:.0%} {side} ({p['mean']:.1f} projected), "
                   f"{verdict}.{hit} Price it on Today's Lines with the real odds before betting.")

    # ---------------------------------------------------------------- cautions
    cautions = []
    if any_r["status"] in ("QUESTIONABLE", "PROBABLE"):
        cautions.append(f"he's listed {any_r['status'].lower()}")
    if any_r["games"] < 10:
        cautions.append(f"only {any_r['games']} games of data this season")
    if any_r["start_rate_last10"] < 0.6:
        cautions.append("he comes off the bench, so his minutes swing more")
    if len(last) >= 5:
        sd = float(np.std(last["min"]))
        if sd > (5.0 if period == "full" else 1.8):
            cautions.append(f"his minutes have bounced around (±{sd:.0f} over the last 10)")
    if any_r["teammates_out"]:
        cautions.append("injury boosts are estimates; check how the coach actually used the lineup "
                        "last time those players sat")
    if cautions:
        out.append("")
        out.append("Be careful: " + "; ".join(cautions) + ".")
    return "\n".join(out)
