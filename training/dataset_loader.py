from pathlib import Path
from typing import List, Optional
import pandas as pd
from config import RAW_DATA_DIR
from utils.logger import logger
from utils.helpers import clean_column_names

class DatasetLoader:
    """Discovers, validates, and loads CIC-IDS-2017 dataset CSV files."""

    def __init__(self, raw_data_dir: Path = RAW_DATA_DIR):
        self.raw_data_dir = Path(raw_data_dir)

    def discover_files(self) -> List[Path]:
        """Lists all CSV files in the raw data directory."""
        files = sorted(list(self.raw_data_dir.glob("*.csv")))
        if not files:
            logger.warning(f"No CSV files found in {self.raw_data_dir}")
        else:
            logger.info(f"Discovered {len(files)} CSV files in {self.raw_data_dir}")
        return files

    def load_file(self, file_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
        """Loads a single CSV file with clean column headers and robust encoding."""
        logger.info(f"Loading {file_path.name} (nrows={nrows})...")
        df = pd.read_csv(file_path, nrows=nrows, encoding="latin-1", low_memory=False)
        df.columns = clean_column_names(df.columns)
        return df

    def load_representative_dataset(
        self,
        samples_per_file: int = 25000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Loads a representative multi-attack dataset from all 8 raw CSV files.
        Extracts rare attack samples completely and samples high-volume classes
        to prevent class imbalance while keeping training time fast and memory-efficient.
        """
        files = self.discover_files()
        if not files:
            raise FileNotFoundError(f"No CSV dataset files found in {self.raw_data_dir}")

        collected_dfs = []
        for file_path in files:
            logger.info(f"Sampling from {file_path.name}...")
            # Read file in chunks or full depending on size
            df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
            df.columns = clean_column_names(df.columns)

            label_col = [c for c in df.columns if c.lower() == "label"][0]
            df[label_col] = df[label_col].astype(str).str.strip()

            # Separate benign and attacks
            benign_mask = df[label_col] == "BENIGN"
            df_benign = df[benign_mask]
            df_attacks = df[~benign_mask]

            # Sample attacks: if small attack count, keep all; if huge (e.g. DDoS/PortScan), sample
            if len(df_attacks) > samples_per_file:
                sampled_attacks = df_attacks.sample(n=samples_per_file, random_state=random_state)
            else:
                sampled_attacks = df_attacks

            # Sample benign to keep ratio reasonable
            n_benign = min(len(df_benign), max(samples_per_file, len(sampled_attacks) * 2))
            sampled_benign = df_benign.sample(n=n_benign, random_state=random_state) if n_benign > 0 else df_benign

            combined = pd.concat([sampled_attacks, sampled_benign], ignore_index=True)
            logger.info(f"Extracted {len(sampled_attacks)} attack rows and {len(sampled_benign)} benign rows from {file_path.name}")
            collected_dfs.append(combined)

        full_df = pd.concat(collected_dfs, ignore_index=True)
        # Shuffle final dataset
        full_df = full_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        logger.info(f"Loaded representative dataset: shape {full_df.shape}")
        return full_df
