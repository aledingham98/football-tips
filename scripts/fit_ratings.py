"""Fit Dixon-Coles team ratings on the full match history and commit them.

    uv run python -m scripts.fit_ratings

Writes data/model/ratings.parquet (+ .json) for the app / pricer to load.
"""

from __future__ import annotations

import pandas as pd

from config.loader import model_config
from models.dixon_coles import DixonColesConfig, fit


def main() -> None:
    matches = pd.read_parquet("data/processed/matches.parquet")
    cfg = DixonColesConfig.from_yaml(model_config())
    ratings = fit(matches, cfg)
    ratings.to_parquet("data/model/ratings.parquet")
    print(f"fitted {ratings.meta['n_teams']} teams on {ratings.meta['n_matches']} matches")
    print(f"  mu={ratings.mu:.3f} gamma={ratings.gamma:.3f} rho={ratings.rho:.3f}")
    print(f"  tier atk means: { {k: round(v, 3) for k, v in ratings.tier_atk.items()} }")
    print(f"  tier dfn means: { {k: round(v, 3) for k, v in ratings.tier_dfn.items()} }")
    # overall strength = attack + defence (both higher = better); spread should span ~+/-0.4
    strength = (ratings.table["atk"] + ratings.table["dfn"]).sort_values(ascending=False)
    print(f"  strength spread: {strength.min():.2f} .. {strength.max():.2f}")
    print("  top 6 / bottom 3:")
    print(strength.head(6).round(3).to_string())
    print(strength.tail(3).round(3).to_string())


if __name__ == "__main__":
    main()
