"""Tunable settings. Change these, then re-run the backtest to see the effect."""

# --- Projection model -------------------------------------------------------
# Everything player-specific uses the current season only.

# Run `python q1.py tune` (or the app's Tune button) on real data to pick these
# from your own results. "q1" = first quarter, "full" = full game.

# Per-minute rates are pulled toward the league average as if the player had
# this many extra minutes at league-average production (fades as minutes pile up).
PRIOR_MINUTES = {"q1": 30.0, "full": 100.0}

# When full-game stats are available, a player's Q1 rate is pulled toward HIS
# OWN full-game rate instead, as if he had this many extra Q1 minutes at it.
# Bigger = trust full games more; smaller = trust his actual Q1s more.
ANCHOR_MINUTES_Q1 = 150.0

# Head-to-head weight = meetings / (meetings + H2H_SHRINK_GAMES[source]).
# Source is full games when downloaded (used for both Q1 and full-game props),
# otherwise Q1 games.  With full=10, a head-to-head ratio gets 9% weight after
# 1 meeting, 17% after 2, 23% after 3, 29% after 4.
H2H_SHRINK_GAMES = {"q1": 20, "full": 10}

# Matchup factor is trusted in proportion n / (n + OPP_SHRINK_GAMES[period]).
# The Q1 factor is shrunk toward the full-game factor (not toward 1.0).
OPP_SHRINK_GAMES = {"q1": 80, "full": 40}
OPP_LOOKBACK_GAMES = 30          # only the opponent's most recent N games

# Players need at least this many appearances this season to get a projection
MIN_GAMES = 3

# --- Recent form (last 10 games) ---------------------------------------------
RECENT_GAMES = 10
RECENT_WEIGHT_MINUTES = 0.5      # minutes = 50% season avg + 50% last-10 avg
RECENT_WEIGHT_RATE = 0.3         # per-minute rate = 70% season + 30% last 10
# last-10 rate is pulled toward his season rate as if he had this many extra minutes
RECENT_PRIOR_MINUTES = {"q1": 20.0, "full": 80.0}

# --- Back-to-backs -----------------------------------------------------------
# Typical effect when a team plays the 2nd night of a back-to-back (own), and when
# its opponent does (opp). The model measures both from your data and moves
# toward the measured value as back-to-back games pile up.
B2B_PRIOR_OWN = 0.97
B2B_PRIOR_OPP = 1.02
B2B_SHRINK_ROWS = 400            # player-games before the data gets 50% weight

# --- Injuries ----------------------------------------------------------------
# A player counts as part of the rotation if he played in the team's last 10 games
# and averages at least this many minutes.
ROTATION_MIN = {"q1": 2.0, "full": 10.0}
MAX_MINUTES = {"q1": 12.0, "full": 40.0}
# Teammates at the same position group (G / F / C) as an injured player get this
# many times more of his minutes than everyone else.
SAME_POSITION_WEIGHT = 2.0
# Share of an injured player's production the rest of the rotation picks up
# (beyond what they produce in their extra minutes).
USAGE_ABSORB = {"pts": 0.5, "reb": 0.3, "ast": 0.5, "fg3m": 0.4, "stl": 0.2, "blk": 0.3}
# Best Spots, 1st quarter: rating = score x minutes weight, where the weight is his
# projected Q1 minutes / Q1_FULL_WEIGHT_MINUTES, kept within Q1_WEIGHT_RANGE.
# 8 minutes = full weight; 10+ = small bonus; 4 minutes = half weight.
Q1_FULL_WEIGHT_MINUTES = 8.0
Q1_WEIGHT_RANGE = (0.5, 1.10)
# --- Simulations -------------------------------------------------------------
SIM_DEFAULT_RUNS = 100           # simulations per matchup / parlay (the app lets you pick more)
HOME_COURT_POINTS = 2.0          # typical home-court edge; the data moves it this season
HOME_COURT_SHRINK_GAMES = 300
# Random swings per game, as a share of a team's expected score:
SIM_PACE_SD = 0.04               # shared by both teams (fast or slow game)
SIM_TEAM_SD = 0.105              # each team's own shooting night
# Team's expected score = (1 - w) x built from player projections + w x its actual
# average this season vs this defense
SIM_TEAM_BASELINE_WEIGHT = 0.4

# How often the app re-downloads the injury report while open (minutes)
INJURY_REFRESH_MINUTES = 30

# Stats the model knows. Combos are sums of the base stats.
BASE_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk"]
COMBOS = {
    "pra": ["pts", "reb", "ast"],
    "pr": ["pts", "reb"],
    "pa": ["pts", "ast"],
    "ra": ["reb", "ast"],
    "sb": ["stl", "blk"],          # "stocks": steals + blocks
}

# --- Betting ----------------------------------------------------------------
BANKROLL = 1000.0
KELLY_FRACTION = 0.25            # quarter Kelly
MAX_STAKE_PCT = 0.02             # never stake more than 2% of bankroll on one bet
MIN_EV = 0.04                    # only flag bets with >= 4% expected value
MIN_EDGE = 0.03                  # and >= 3 pts of probability over the no-vig price

# --- Data fetching ----------------------------------------------------------
REQUEST_PAUSE_SEC = 0.7          # stats.nba.com throttles fast clients
REQUEST_TIMEOUT_SEC = 30
REQUEST_RETRIES = 3
