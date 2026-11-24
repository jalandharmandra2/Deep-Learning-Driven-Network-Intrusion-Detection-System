import json
from pathlib import Path
from typing import List
import pandas as pd
from config import FEATURE_COLUMNS, FEATURE_COLUMNS_PATH
from utils.logger import logger
from utils.helpers import clean_column_names

class FeatureSelector:
    """Enforces the 52 validated feature schema contract."""

    def __init__(self, target_features: List[str] = FEATURE_COLUMNS):
        self.target_features = list(target_features)

    def select_features(self, df: pd.DataFrame, fill_missing: bool = True) -> pd.DataFrame:
        """
        Selects and reorders columns strictly to match target_features (52 features).
        If fill_missing is True, any missing feature is populated with 0.0 with a warning.
        """
        df.columns = clean_column_names(df.columns)
        missing_cols = [c for c in self.target_features if c not in df.columns]

        if missing_cols:
            if fill_missing:
                logger.warning(
                    f"{len(missing_cols)} features missing in input data. Filling with 0.0: {missing_cols}"
                )
                for col in missing_cols:
                    df[col] = 0.0
            else:
                raise KeyError(f"Input DataFrame is missing required features: {missing_cols}")

        # Return strictly the 52 features in exact order
        selected_df = df[self.target_features].copy()
        return selected_df

    def save_feature_schema(self, path: Path = FEATURE_COLUMNS_PATH) -> None:
        """Saves feature columns schema to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "feature_count": len(self.target_features),
                "features": self.target_features
            }, f, indent=4)
        logger.info(f"Saved feature schema ({len(self.target_features)} features) to {path}")

    @classmethod
    def load_feature_schema(cls, path: Path = FEATURE_COLUMNS_PATH) -> "FeatureSelector":
        """Loads feature columns schema from JSON."""
        if not path.exists():
            logger.warning(f"{path} does not exist. Using default FEATURE_COLUMNS.")
            return cls(FEATURE_COLUMNS)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["features"])
