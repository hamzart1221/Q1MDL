"""Read parlay legs from text copied out of a sportsbook / pick'em app.

Handles formats like:
    Paul Reed  Under 1.5 Blocks
    Jalen Johnson Over 34.5 Pts + Rebs + Asts
    Jalen Brunson NYK - G  24.5 Points  More           (PrizePicks style)
    Shai Gilgeous-Alexander 30+ Points                 (alt-line style = over 29.5)
    Tyrese Maxey 1Q Over 7.5 Points / 1st Quarter Points
    Knicks ML / Knicks Moneyline / Knicks -4.5 / Pacers @ Knicks Over 229.5
and the payout: "~2.42x", "$10 pays $24.20", "+450".
"""
from __future__ import annotations

import re

from .feeds import TEAM_NAMES, team_abbr
from .model import norm_name

# longest phrases first so combos win over single stats
STAT_PHRASES = [
    (r"pts\s*\+\s*rebs?\s*\+\s*asts?|points\s*\+\s*rebounds\s*\+\s*assists|p\s*\+\s*r\s*\+\s*a|\bpra\b", "pra"),
    (r"pts\s*\+\s*rebs?|points\s*\+\s*rebounds|\bp\s*\+\s*r\b|\bpr\b", "pr"),
    (r"pts\s*\+\s*asts?|points\s*\+\s*assists|\bp\s*\+\s*a\b|\bpa\b", "pa"),
    (r"rebs?\s*\+\s*asts?|rebounds\s*\+\s*assists|\br\s*\+\s*a\b|\bra\b", "ra"),
    (r"blks?\s*\+\s*stls?|stls?\s*\+\s*blks?|steals\s*\+\s*blocks|blocks\s*\+\s*steals|stocks", "sb"),
    (r"3[\s-]*pointers?(\s*made)?|3[\s-]*pt\w*(\s*made)?|three[\s-]*pointers?(\s*made)?|threes|\b3pm\b|\bfg3m\b", "fg3m"),
    (r"\bpoints?\b|\bpts\b", "pts"),
    (r"\brebounds?\b|\brebs?\b", "reb"),
    (r"\bassists?\b|\basts?\b", "ast"),
    (r"\bsteals?\b|\bstls?\b", "stl"),
    (r"\bblocks?\b|\bblks?\b|blocked shots", "blk"),
]
OVER = r"\b(over|more|higher|o)\b|\bo(?=\d)"
UNDER = r"\b(under|less|lower|u)\b|\bu(?=\d)"
Q1 = r"\b(1q|1st\s*q(uarter)?|first\s*quarter|q1)\b"
ALIASES = {"sga": "Shai Gilgeous-Alexander", "kat": "Karl-Anthony Towns", "ad": "Anthony Davis",
           "jjj": "Jaren Jackson Jr.", "pg13": "Paul George"}


def _payout(text: str):
    """Book's parlay odds as American odds (string) if the text shows them."""
    t = text.lower().replace(",", "")
    m = re.search(r"\$\s*([\d.]+)\s*pays\s*~?\s*\$\s*([\d.]+)", t)
    if m and float(m.group(1)) > 0:
        mult = float(m.group(2)) / float(m.group(1))
    else:
        m = re.search(r"~?\s*([\d]+(?:\.\d+)?)\s*x\b", t)
        mult = float(m.group(1)) if m else None
    if mult and mult > 1.01:
        return f"+{round((mult - 1) * 100)}" if mult >= 2 else f"{round(-100 / (mult - 1))}"
    m = re.search(r"(?:odds|parlay|payout)[^\n+-]*([+-]\d{3,5})\b", t)
    return m.group(1) if m else None


def _team_mentions(seg: str) -> list[str]:
    found = []
    low = seg.lower()
    for ab, name in TEAM_NAMES.items():
        nick = name.split()[-1].lower()
        if re.search(rf"\b{re.escape(nick)}\b", low) or re.search(rf"\b{ab.lower()}\b", low):
            found.append((low.find(nick) if nick in low else low.find(ab.lower()), ab))
    if re.search(r"\btrail blazers\b|\bblazers\b", low) and "POR" not in [a for _, a in found]:
        found.append((low.find("blazers"), "POR"))
    return [ab for _, ab in sorted(found)]


class Parser:
    def __init__(self, players: dict[str, str], tonight_teams: set[str] | None = None):
        """players: display name -> team abbreviation (everyone the model knows)."""
        self.players = players
        self.tonight = tonight_teams or set()
        self.by_key = {norm_name(n): n for n in players}
        self.by_last: dict[str, list[str]] = {}
        for n in players:
            parts = norm_name(n).split()
            if parts:
                self.by_last.setdefault(parts[-1], []).append(n)

    # ------------------------------------------------------------ players
    def _find_players(self, text: str) -> list[tuple[int, int, str]]:
        """(start, end, name) for every player mentioned, in order."""
        norm = text.lower()
        hits = []
        for alias, name in ALIASES.items():
            for m in re.finditer(rf"\b{alias}\b", norm):
                if name in self.players:
                    hits.append((m.start(), m.end(), name))
        # full names, matched on letters only so accents / punctuation / Jr. don't matter
        for line_m in re.finditer(r"[^\n]+", text):
            line = line_m.group(0)
            words = list(re.finditer(r"[A-Za-zÀ-ÿ'’.\-]+", line))
            keys = [norm_name(w.group(0)) for w in words]
            i = 0
            while i < len(words):
                matched = False
                for span in (4, 3, 2):
                    if i + span > len(words):
                        continue
                    key = " ".join(k for k in keys[i:i + span] if k)
                    if key in self.by_key:
                        hits.append((line_m.start() + words[i].start(),
                                     line_m.start() + words[i + span - 1].end(), self.by_key[key]))
                        i += span
                        matched = True
                        break
                if not matched:
                    # "Cam Johnson" -> Cameron Johnson: same last name, same first initial
                    if i + 1 < len(words) and keys[i] and keys[i + 1] in self.by_last:
                        same_last = self.by_last[keys[i + 1]]
                        cands = [n for n in same_last if norm_name(n).split()[0].startswith(keys[i])]
                        if not cands:  # fall back to first initial
                            cands = [n for n in same_last if norm_name(n).split()[0][:1] == keys[i][:1]]
                        if self.tonight:
                            on = [n for n in cands if self.players[n] in self.tonight]
                            cands = on or cands
                        if len(cands) == 1 and keys[i] not in ("over", "under", "more", "less"):
                            hits.append((line_m.start() + words[i].start(),
                                         line_m.start() + words[i + 1].end(), cands[0]))
                            i += 2
                            continue
                    i += 1
        hits.sort()
        out, last_end = [], -1
        for h in hits:   # drop overlaps
            if h[0] >= last_end:
                out.append(h)
                last_end = h[1]
        return out

    # ------------------------------------------------------------ parse
    def _prop(self, name: str, seg: str) -> tuple[dict | None, str | None]:
        low = seg.lower()
        stat = next((key for pat, key in STAT_PHRASES if re.search(pat, low)), None)
        if not stat:
            return None, f"{name}: couldn't tell which stat"
        period = "q1" if re.search(Q1, low) else "full"
        m_plus = re.search(r"(\d+)\s*\+", low)
        nums = [float(x) for x in re.findall(r"(?<![\d$])(\d+(?:\.\d+)?)(?![\d.]*\s*x\b)", low)]
        nums = [n for n in nums if n not in (1.0,) or "1q" not in low]  # don't read "1Q" as a line
        if re.search(Q1, low):
            nums = [n for n in nums if not (n == 1.0 and re.search(r"\b1\s*(q|st)", low))]
        if m_plus and not re.search(OVER + "|" + UNDER, low):
            return dict(type="prop", player=name, stat=stat, period=period, side="over",
                        line=float(m_plus.group(1)) - 0.5), None
        if not nums:
            return None, f"{name}: couldn't find the line"
        line = nums[0]
        if re.search(UNDER, low):
            side = "under"
        elif re.search(OVER, low):
            side = "over"
        else:
            return None, f"{name} {line:g} {stat}: couldn't tell over or under"
        return dict(type="prop", player=name, stat=stat, period=period, side=side, line=line), None

    def _team_legs(self, seg: str) -> tuple[list[dict], list[str]]:
        legs, problems = [], []
        for raw in [l for l in re.split(r"[\n;|]", seg) if l.strip()]:
            low = raw.lower()
            teams = _team_mentions(raw)
            if not teams:
                continue
            if re.search(OVER + "|" + UNDER, low) and re.search(r"\d{3}(\.\d)?", low):
                line = float(re.search(r"(\d{3}(?:\.\d)?)", low).group(1))
                side = "under" if re.search(UNDER, low) else "over"
                legs.append(dict(type="total", team=teams[0], side=side, line=line))
            elif re.search(r"\bml\b|money\s*line|to win\b|\bwins?\b", low):
                legs.append(dict(type="moneyline", team=teams[-1] if " to win" not in low else teams[0]))
            else:
                m = re.search(r"(?<![\d.])([+-]\d{1,2}(?:\.\d)?)(?!\d)", low)
                if m:
                    legs.append(dict(type="spread", team=teams[0], line=float(m.group(1))))
        return legs, problems

    def parse(self, text: str) -> dict:
        text = text.replace("\u2212", "-").replace("–", "-")
        hits = self._find_players(text)
        legs, problems = [], []
        used = [False] * len(text)
        has_line = re.compile(r"\d")
        for i, (start, end, name) in enumerate(hits):
            limit = hits[i + 1][0] if i + 1 < len(hits) else len(text)
            # the rest of the player's line, plus up to 2 more lines if the stat/line isn't there yet
            seg_end = text.find("\n", end)
            seg_end = limit if seg_end == -1 or seg_end > limit else seg_end
            for _ in range(2):
                piece = text[end:seg_end].lower()
                if has_line.search(piece) and any(re.search(p, piece) for p, _ in STAT_PHRASES):
                    break
                nxt = text.find("\n", seg_end + 1)
                nxt = limit if nxt == -1 or nxt > limit else nxt
                if nxt <= seg_end:
                    break
                seg_end = nxt
            seg = text[end:seg_end]
            leg, err = self._prop(name, seg)
            if not leg:  # maybe the line came before the name ("Over 22.5 points Jalen Brunson")
                line_start = text.rfind("\n", 0, start) + 1
                prev_end = hits[i - 1][1] if i > 0 else 0
                pre_start = max(line_start, prev_end)
                leg2, _ = self._prop(name, text[pre_start:start] + " " + seg)
                if leg2:
                    leg, err = leg2, None
                    start = pre_start
            for k in range(start, seg_end):
                used[k] = True
            if leg:
                legs.append(leg)
            else:
                problems.append(err)
        rest = "".join(ch if not used[k] else ("\n" if ch == "\n" else " ") for k, ch in enumerate(text))
        tl, tp = self._team_legs(rest)
        legs += tl
        problems += tp
        seen, uniq = set(), []
        for l in legs:
            k = tuple(sorted(l.items()))
            if k not in seen:
                seen.add(k)
                uniq.append(l)
        return {"legs": uniq, "problems": [p for p in problems if p], "odds": _payout(text)}
