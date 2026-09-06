# football-tips

Personal betting-analysis tool for the English Premier League, Championship, League One
and League Two. Its job is **pricing bet builders (same-game multis) and accumulators** so a
price can be checked for value before it is taken.

Personal use, UK user, zero running cost, deployed on Streamlit Community Cloud.

---

## Core idea

One **Monte Carlo match simulator**, not a model per market. Each simulation produces a
complete synthetic match — final score, shots and shots on target per player, goalscorers and
assisters, cards per player, corners per team, and minute-level goal timing. Any market is then
a **count over simulations**:

- `Over 2.5 goals` = fraction of sims with 3+ goals
- `Player X 2+ SoT` = fraction of sims where X recorded 2+ shots on target
- `Home win AND BTTS AND Player X 2+ SoT` = fraction of sims where **all** hold at once

Bet-builder legs are correlated, so the joint probability is counted over a shared simulation,
never obtained by multiplying leg probabilities. Accumulator legs sit in separate matches, so
they multiply — with the compounding bookmaker margin shown explicitly.

## Architecture

```
GitHub Actions (scheduled)                Streamlit Cloud (redeploys on push)
  nightly-data.yml   FBref + fixtures,      app reads data/*.parquet only
                     refit Dixon-Coles,     app reads Supabase for the bet log
                     write + commit parquet app calls NO external API at runtime
  odds-refresh.yml   The Odds API (budgeted) + Betfair delayed -> commit parquet
```

The app never calls FBref or an odds API at request time. Everything it needs is committed to
the repo as parquet by the scheduled jobs; the bet log lives in Supabase because Streamlit
Cloud has an ephemeral filesystem.

## Layout

| Dir           | Purpose                                                              |
| ------------- | ------------------------------------------------------------------ |
| `ingest/`     | FBref (soccerdata), football-data.org, football-data.co.uk, odds providers |
| `models/`     | Dixon-Coles fit, league strength, minutes / shots / cards / corners sub-models |
| `simulation/` | Vectorised Monte Carlo engine + market counting (no Streamlit import) |
| `pricing/`    | EV, edge, no-vig, Kelly, CLV, bet-builder and acca pricing         |
| `storage/`    | Runtime-writable store (Supabase) for the bet log and manual odds  |
| `app/`        | Streamlit pages (mobile-first)                                     |
| `config/`     | League maps, model hyperparameters, market catalogue              |
| `data/`       | `processed/` + `model/` parquet committed by CI; raw caches gitignored |

## Setup

```bash
uv sync --extra dev --extra ingest --extra backtest   # create .venv (Python 3.12)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill in keys
uv run pytest                                          # engine + fit + pricing tests
uv run python -m ingest.matches                        # build data/processed/matches.parquet
uv run python -m scripts.fit_ratings                   # -> data/model/ratings.parquet
uv run python -m scripts.run_backtest                  # -> reports/calibration/reliability.png
uv run streamlit run app/Home.py                       # the app, on http://localhost:8501
```

Regenerate the pinned deploy manifests after changing dependencies (keep the `-e .` line):

```bash
uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt
uv export --no-hashes --no-dev --extra ingest --format requirements-txt -o requirements-ingest.txt
```

## Deploy to Streamlit Community Cloud

1. Push to GitHub (done: `aledingham98/football-tips`, public).
2. At [share.streamlit.io](https://share.streamlit.io) → **New app** → this repo, branch `main`,
   main file `app/Home.py`. It reads `.python-version` (3.12) and installs `requirements.txt`
   (which includes `-e .` so the local packages resolve).
3. **Settings → Secrets**: paste the filled-in contents of `.streamlit/secrets.toml.example`.
   The app itself needs no secret in Phase 3 (it only reads committed parquet); the keys are
   for later phases and parity with the Action.
4. The nightly `nightly-data` GitHub Action refreshes the parquet and commits it back; Streamlit
   Cloud auto-redeploys on each push. Add `FOOTBALL_DATA_ORG_API_KEY` as a repo secret for it.

## Build status

- [x] **Phase 1** — Dixon-Coles fit, vectorised simulation engine, tests (50k sims ≈ 0.3 s)
- [x] **Phase 2** — walk-forward calibration on a held-out season; 1X2 calibrated, goals-markets
      tuning tracked (tempo term + calibration map in place, xG via the Action pending)
- [~] **Phase 3** — Bet Builder pricer UI built; Streamlit Cloud deploy is a one-time account step
- [ ] **Phase 4** — Value / High Confidence lists, The Odds API + Betfair
- [ ] **Phase 5** — Stats browser, acca builder, bet log

### Data note

football-data.org's free tier gives Premier League + Championship, three seasons. The nightly
Action also tries football-data.co.uk (four tiers, five seasons, closing odds + referee) and
FBref via `soccerdata` (player-level, all four tiers); both can be flaky from cloud IPs, so the
pipeline degrades gracefully and the app always has *something* to price.

## Not a tipster

Bet builders carry 15–30% margin. The realistic job is filtering out bad prices. An empty
value list is correct behaviour. If the tool surfaces dozens of value bets per matchweek, the
model is broken. Where the model and a liquid Betfair market disagree sharply, that is
surfaced as possible model error, not as a betting opportunity.
