import numpy as np
import pandas as pd
from typing import List, Optional
from utils.logger import logger
from utils.helpers import clean_column_names

class DataCleaner:
    """Handles missing values, infinite numbers, and deduplication."""

    @staticmethod
    def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Strips whitespace from column names."""
        df.columns = clean_column_names(df.columns)
        return df

    @staticmethod
    def handle_infinite(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Replaces infinite values (+inf, -inf) with upper/lower finite limits or zero.
        Specifically addresses Flow Bytes/s and Flow Packets/s in CIC-IDS-2017.
        """
        target_cols = columns if columns is not None else [
            c for c in df.columns if c != "Label" and df[c].dtype != object
        ]

        for col in target_cols:
            if col in df.columns:
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
                # Check for inf
                inf_mask = np.isinf(df[col])
                if inf_mask.any():
                    # Compute max finite value
                    finite_vals = df.loc[~inf_mask, col].dropna()
                    if len(finite_vals) > 0:
                        max_val = finite_vals.quantile(0.999)  # 99.9th percentile to prevent outlier skew
                        df.loc[df[col] > max_val, col] = max_val
                        df.loc[df[col] < 0, col] = 0
                    else:
                        df[col] = df[col].replace([np.inf, -np.inf], 0)
        return df

    @staticmethod
    def handle_missing(df: pd.DataFrame, strategy: str = "zero") -> pd.DataFrame:
        """Imputes missing (NaN) values."""
        numeric_cols = [c for c in df.columns if c != "Label" and df[c].dtype != object]
        if strategy == "zero":
            df[numeric_cols] = df[numeric_cols].fillna(0.0)
        elif strategy == "median":
            for col in numeric_cols:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)
        return df

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Removes duplicate rows while logging count."""
        initial_len = len(df)
        df = df.drop_duplicates(subset=subset).reset_index(drop=True)
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"Removed {dropped} duplicate rows ({initial_len} -> {len(df)})")
        return df

    def clean(self, df: pd.DataFrame, target_features: Optional[List[str]] = None) -> pd.DataFrame:
        """Executes the full cleaning pipeline on a DataFrame."""
        df = self.clean_columns(df)
        df = self.handle_infinite(df, columns=target_features)
        df = self.handle_missing(df, strategy="zero")
        df = self.remove_duplicates(df, subset=target_features)
        return df
