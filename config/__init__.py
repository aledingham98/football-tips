"""YAML config loaders. ``config/*.yaml`` sit next to this package."""

from config.loader import league_table, leagues_config, model_config, tier_of

__all__ = ["leagues_config", "model_config", "league_table", "tier_of"]
