"""Q1 Model - phone / web version (runs on Streamlit Community Cloud, no PC needed).

Stats are refreshed every morning by the GitHub Action in .github/workflows/update-stats.yml
(ESPN feeds -> data/). Injuries and tonight's schedule are pulled live from ESPN.
Open the app link on your phone and use "Add to Home Screen" to get an app icon.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from q1model.core import STAT_LABELS, Core

ROOT = Path(__file__).parent
DATA = ROOT / "data"
HOME = ROOT / ".cloud_home"
ET = timezone(timedelta(hours=-5))

st.set_page_config(page_title="Q1 Model", page_icon="🏀", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container{padding-top:1rem;padding-left:.7rem;padding-right:.7rem;max-width:900px}
.hero{background:linear-gradient(90deg,#1d428a,#c8102e);border-radius:14px;padding:12px 14px;color:#fff;margin-bottom:8px}
.hero h2{margin:0;font-size:1.3rem}.hero p{margin:.2rem 0 0;opacity:.85;font-size:.85rem}
.pk{border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:9px 11px;margin:7px 0}
.pk .top{display:flex;justify-content:space-between;font-weight:700;font-size:1.02rem}
.ov{color:#1fb866}.un{color:#e8475b}.mu{opacity:.7;font-size:.8rem}
pre{font-size:11px !important;line-height:1.3 !important;white-space:pre-wrap !important}
.stTabs [data-baseweb="tab-list"]{gap:2px;overflow-x:auto}
.stTabs [data-baseweb="tab"]{padding:6px 9px;font-size:.85rem}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model (cached)
def _data_stamp() -> float:
    f = DATA / "full_player_games.csv"
    return f.stat().st_mtime if f.exists() else 0.0


@st.cache_resource(show_spinner="Loading stats...")
def get_core(stamp: float) -> Core:
    HOME.mkdir(exist_ok=True)
    c = Core(home=HOME, data_dir=DATA)
    c.load_model()
    try:
        c.refresh_injuries()
    except Exception:  # noqa: BLE001 - app still works without the live report
        pass
    return c


core = get_core(_data_stamp())
if core.model is None:
    st.error("No stats yet. On GitHub open Actions → 'Update stats' → Run workflow, wait ~10 minutes, "
             "then reload this page.")
    st.stop()


@st.cache_data(ttl=1800, show_spinner=False)
def refresh_injuries(_bucket: int) -> str:
    try:
        return core.refresh_injuries()
    except Exception as e:  # noqa: BLE001
        return f"Injury refresh failed: {e}"


inj_msg = refresh_injuries(int(datetime.now().timestamp() // 1800))   # every 30 minutes

# ------------------------------------------------------------------ header + date
today = datetime.now(ET).date()
last = core.table["game_date"].max() if core.table is not None and len(core.table) else None
day = st.date_input("Slate date", today, label_visibility="collapsed")
st.markdown(f"<div class='hero'><h2>🏀 Q1 Model · {day:%a %b %d}</h2>"
            f"<p>Stats through {last:%b %d} · {inj_msg}</p></div>", unsafe_allow_html=True)

STAT_KEYS = list(STAT_LABELS)
tabs = st.tabs(["★ Picks", "🔍 Lookup", "📈 Best spots", "🎲 Parlay", "🏟 Games", "🩹 Injuries"])


def stat_name(k: str) -> str:
    return STAT_LABELS.get(k, k)


# ------------------------------------------------------------------ ★ favorite picks
@st.cache_data(ttl=1800, show_spinner="Finding today's picks...")
def favs(day_s: str, stamp: float, _bucket: int):
    return core.favorite_picks(day_s)


with tabs[0]:
    try:
        fp, msg = favs(str(day), _data_stamp(), int(datetime.now().timestamp() // 1800))
    except Exception as e:  # noqa: BLE001
        fp, msg = pd.DataFrame(), str(e)
    st.caption(f"{msg}. Over = highest line he clears ~60% of the time; Under = lowest line he stays under "
               "~60%. A better book line is even better.")
    if len(fp):
        side = st.radio("Show", ["Overs", "Unders"], horizontal=True, label_visibility="collapsed")
        sub = fp[fp["side"] == ("over" if side == "Overs" else "under")]
        cls = "ov" if side == "Overs" else "un"
        for r in sub.itertuples():
            st.markdown(
                f"<div class='pk'><div class='top'><span class='{cls}'>{r.side[0].upper()}{r.rank}. {r.player}"
                f"</span><span>{r.chance:.0%}</span></div><div class='top'><span class='{cls}'>{r.pick}</span>"
                f"<span class='mu'>proj {r.proj:.1f} · avg {r.season:.1f}</span></div>"
                f"<div class='mu'>{r.team} vs {r.opp} · {r.why}</div></div>", unsafe_allow_html=True)
        if st.button(f"➕ Add these {side.lower()} to the parlay", use_container_width=True):
            st.session_state.setdefault("legs", [])
            for r in sub.itertuples():
                st.session_state["legs"].append(dict(type="prop", player=r.player, stat=r.stat, period="full",
                                                     side=r.side, line=float(r.target)))
            st.success("Added - open the 🎲 Parlay tab.")


# ------------------------------------------------------------------ 🔍 lookup
def tonight_opp(team: str) -> str:
    try:
        games, _ = core.schedule(day)
    except Exception:  # noqa: BLE001
        return ""
    for g in games.itertuples():
        if g.away == team:
            return g.home
        if g.home == team:
            return g.away
    return ""


with tabs[1]:
    names = sorted(core.player_names())
    player = st.selectbox("Player", names, index=None, placeholder="Type a name...")
    if player:
        team = core.player_team(player)
        teams = sorted(core.model.directory["team"].dropna().unique())
        opp_guess = tonight_opp(team)
        c1, c2 = st.columns(2)
        opp = c1.selectbox("Opponent", teams, index=teams.index(opp_guess) if opp_guess in teams else 0)
        period = c2.radio("Period", ["Full game", "1st quarter"], horizontal=True)
        c3, c4 = st.columns(2)
        stat = c3.selectbox("Stat for the line", STAT_KEYS, format_func=stat_name)
        line_txt = c4.text_input("Line (optional)", "")
        try:
            line = float(line_txt) if line_txt.strip() else None
        except ValueError:
            line, _ = None, st.warning("Line must be a number like 24.5")
        if not opp_guess:
            st.caption(f"{team} isn't on the {day:%b %d} slate - pick the opponent.")
        try:
            txt = core.lookup(player, opp, line, period="q1" if period == "1st quarter" else "full",
                              line_stat=stat, day=day)
            st.code(txt, language=None)
        except Exception as e:  # noqa: BLE001
            st.warning(str(e))


# ------------------------------------------------------------------ 📈 best spots
@st.cache_data(ttl=1800, show_spinner="Ranking tonight...")
def spots(day_s: str, stamp: float, _bucket: int):
    return core.best_spots_all(day_s)


with tabs[2]:
    try:
        bs, msg = spots(str(day), _data_stamp(), int(datetime.now().timestamp() // 1800))
    except Exception as e:  # noqa: BLE001
        bs, msg = pd.DataFrame(), str(e)
    st.caption(msg)
    if len(bs):
        c1, c2 = st.columns(2)
        per = c1.radio("Period", ["Full game", "1st quarter"], horizontal=True, key="bs_per")
        cats = ["All"] + sorted(bs["category"].dropna().unique())
        cat = c2.selectbox("Category", cats)
        view = bs[bs["period"] == ("q1" if per == "1st quarter" else "full")]
        if cat != "All":
            view = view[view["category"] == cat]
        view = view.assign(Stat=view["stat"].map(stat_name))
        st.dataframe(view[["rating", "player", "team", "opp", "Stat", "proj", "season", "tier", "why"]]
                     .rename(columns={"rating": "Rating", "player": "Player", "team": "Team", "opp": "Opp",
                                      "proj": "Proj", "season": "Avg", "tier": "Tier", "why": "Why"}),
                     hide_index=True, use_container_width=True, height=520,
                     column_config={"Rating": st.column_config.NumberColumn(format="%.2f"),
                                    "Proj": st.column_config.NumberColumn(format="%.1f"),
                                    "Avg": st.column_config.NumberColumn(format="%.1f")})


# ------------------------------------------------------------------ 🎲 parlay
with tabs[3]:
    st.session_state.setdefault("legs", [])
    pasted = st.text_area("Paste your legs (one per line)", height=110,
                          placeholder="Buddy Hield Over 13.5 Points\nKeyonte George 30 PA over\nKnicks ML")
    c1, c2 = st.columns(2)
    if c1.button("📋 Read & add legs", use_container_width=True) and pasted.strip():
        try:
            res = core.parse_legs(pasted, day)
            st.session_state["legs"] += res["legs"]
            if res.get("odds"):
                st.session_state["odds"] = str(res["odds"])
            for p in res.get("problems", []):
                st.warning(p)
        except Exception as e:  # noqa: BLE001
            st.warning(str(e))
    if c2.button("🗑 Clear legs", use_container_width=True):
        st.session_state["legs"] = []

    legs = st.session_state["legs"]
    if legs:
        st.caption("Edit any leg right in the table (side, line, stat, period). Untick 'Keep' to remove one.")
        df = pd.DataFrame([dict(Keep=True, Type=l["type"], Who=l.get("player") or l.get("team", ""),
                                Stat=l.get("stat", ""), Period=l.get("period", ""),
                                Side=l.get("side", ""), Line=l.get("line")) for l in legs])
        ed = st.data_editor(
            df, hide_index=True, use_container_width=True, key="legs_editor",
            column_config={
                "Keep": st.column_config.CheckboxColumn(width="small"),
                "Type": st.column_config.TextColumn(disabled=True),
                "Stat": st.column_config.SelectboxColumn(options=[""] + STAT_KEYS),
                "Period": st.column_config.SelectboxColumn(options=["", "full", "q1"]),
                "Side": st.column_config.SelectboxColumn(options=["", "over", "under"]),
                "Line": st.column_config.NumberColumn(step=0.5)})
        new_legs = []
        for (_, row), old in zip(ed.iterrows(), legs):
            if not row["Keep"]:
                continue
            leg = dict(old)
            if leg["type"] == "prop":
                leg.update(player=row["Who"], stat=row["Stat"] or leg["stat"], period=row["Period"] or "full",
                           side=row["Side"] or "over", line=float(row["Line"]) if pd.notna(row["Line"]) else 0.5)
            elif leg["type"] in ("spread", "total"):
                leg.update(team=row["Who"], line=float(row["Line"]) if pd.notna(row["Line"]) else leg.get("line"))
                if leg["type"] == "total":
                    leg["side"] = row["Side"] or leg.get("side", "over")
            else:
                leg.update(team=row["Who"])
            new_legs.append(leg)

        c1, c2 = st.columns(2)
        odds = c1.text_input("Book's parlay odds (optional)", st.session_state.get("odds", ""),
                             placeholder="+600 or 6x")
        n = c2.select_slider("Simulations", [1000, 10000, 25000], value=10000)
        if st.button("🎲 Simulate parlay", type="primary", use_container_width=True):
            try:
                with st.spinner("Simulating..."):
                    r = core.simulate_parlay(new_legs, day, n, odds.strip() or None)
                st.markdown(f"### Hit {r['hits']:,} of {r['sims']:,} times · **{r['prob']:.1%}**")
                st.caption(f"±{r['moe']:.1%} · fair odds {r['fair_odds']} · if the legs were unrelated: "
                           f"{r['independent_prob']:.1%}")
                if "ev" in r:
                    verdict = ("positive expected value ✅" if r["ev"] > 0.03 else
                               "about break-even" if r["ev"] > -0.03 else "negative expected value ❌")
                    st.markdown(f"Book odds imply **{r['book_implied']:.1%}** → EV **{r['ev']:+.1%}** per $1 "
                                f"({verdict})")
                st.dataframe(pd.DataFrame([dict(Leg=l["leg"], Hit=f"{l['hits']:,}", Chance=f"{l['prob']:.1%}",
                                                Fair=l["fair"]) for l in r["legs"]]),
                             hide_index=True, use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.warning(str(e))
        with st.expander("🔍 Look up a leg's player"):
            props = [l for l in new_legs if l["type"] == "prop"]
            if props:
                pick = st.selectbox("Leg", range(len(props)),
                                    format_func=lambda i: f"{props[i]['player']} {props[i]['side']} "
                                                          f"{props[i]['line']:g} {stat_name(props[i]['stat'])}")
                l = props[pick]
                opp = tonight_opp(core.player_team(l["player"]))
                if opp:
                    st.code(core.lookup(l["player"], opp, l["line"], period=l["period"], line_stat=l["stat"],
                                        day=day), language=None)
                else:
                    st.caption("His team isn't on this date's slate.")
    else:
        st.caption("No legs yet - paste some above, or add picks from the ★ Picks tab.")


# ------------------------------------------------------------------ 🏟 games
@st.cache_data(ttl=1800, show_spinner="Simulating tonight's games...")
def game_sims(day_s: str, stamp: float, _bucket: int):
    return core.simulate_games(day_s, 1000)


with tabs[4]:
    try:
        g, msg = game_sims(str(day), _data_stamp(), int(datetime.now().timestamp() // 1800))
    except Exception as e:  # noqa: BLE001
        g, msg = pd.DataFrame(), str(e)
    st.caption(msg)
    for r in g.itertuples() if len(g) else []:
        fav = r.home if r.home_win >= r.away_win else r.away
        st.markdown(
            f"<div class='pk'><div class='top'><span>{r.away} @ {r.home}</span><span class='mu'>{r.tip}</span></div>"
            f"<div>{r.away} {r.away_win:.0%} ({r.away_fair}) · {r.home} {r.home_win:.0%} ({r.home_fair})</div>"
            f"<div class='mu'>Proj {r.away_score:.0f}-{r.home_score:.0f} · {fav} favored · spread "
            f"{r.home} {r.home_spread:+.1f} · total {r.total:.1f}"
            + (f" · out: {r.away_out} / {r.home_out}" if (r.away_out or r.home_out) else "")
            + "</div></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ 🩹 injuries
with tabs[5]:
    if st.button("🔄 Refresh injuries now"):
        st.info(core.refresh_injuries())
        st.cache_data.clear()
    inj = core.injury_table()
    if len(inj):
        teams = ["All"] + sorted(inj["team"].dropna().unique())
        t = st.selectbox("Team", teams)
        st.dataframe(inj if t == "All" else inj[inj["team"] == t], hide_index=True, use_container_width=True)

st.caption("For entertainment and research. Bet responsibly.")
