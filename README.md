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
uv sync --extra dev          # create .venv (Python 3.12) and install
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill in keys
uv run pytest                # run the engine test suite
```

Regenerate the pinned deploy manifest after changing dependencies:

```bash
uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt
```

## Build status

- [ ] **Phase 1** — FBref ingestion, Dixon-Coles fit, simulation engine, tests
- [ ] **Phase 2** — backtest + calibration on a held-out season
- [ ] **Phase 3** — Bet Builder pricer UI, deployed to Streamlit Cloud
- [ ] **Phase 4** — Value / High Confidence lists, The Odds API + Betfair
- [ ] **Phase 5** — Stats browser, acca builder, bet log

## Not a tipster

Bet builders carry 15–30% margin. The realistic job is filtering out bad prices. An empty
value list is correct behaviour. If the tool surfaces dozens of value bets per matchweek, the
model is broken. Where the model and a liquid Betfair market disagree sharply, that is
surfaced as possible model error, not as a betting opportunity.
