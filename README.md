# Q1 Model - phone app

The Q1 / full-game props model as a web app you open on your phone. Runs in the cloud, no PC needed.

* `streamlit_app.py` - the app (Favorite Picks, Player Lookup, Best Spots, Parlay sim, Game sims, Injuries)
* `update_stats.py` + `.github/workflows/update-stats.yml` - downloads new games from ESPN every morning into `data/`
* `q1model/` - the same model as the desktop app

## One-time setup (about 15 minutes, do it in any web browser)
1. **GitHub:** sign up at github.com (free). Click **+ → New repository**, name it `q1-model`, click **Create repository**.
2. On the new repo page click **uploading an existing file**. Drag in *everything inside* the `q1-model-phone`
   folder (including the `.github` and `.streamlit` folders). Click **Commit changes**.
3. Open the **Actions** tab → click **I understand, enable** if asked → **Update stats** → **Run workflow**.
   The first run downloads the whole season (15-25 min). When it shows a green check, stats are in.
   After that it runs by itself every morning.
4. **Streamlit:** go to share.streamlit.io, sign in **with GitHub**, click **Create app → Deploy a public app from GitHub**.
   Pick repository `q1-model`, branch `main`, main file `streamlit_app.py`, and choose a URL (e.g. `hamza-q1`). Click **Deploy**.
5. **iPhone:** open your app link in Safari → Share button → **Add to Home Screen**. Done - it opens like an app.

If the app hasn't been opened for a few days it naps; tap "Yes, get this app back up" and wait ~30 seconds.
