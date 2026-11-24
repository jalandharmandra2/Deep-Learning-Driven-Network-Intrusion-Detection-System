from pathlib import Path
from typing import List, Tuple, Any
import pandas as pd
from config import FEATURE_COLUMNS

class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass

class DataValidator:
    """Validates input datasets, files, and feature vectors."""

    @staticmethod
    def validate_file_extension(file_path: Path, allowed_extensions: Tuple[str, ...]) -> bool:
        """Checks if a file extension is allowed."""
        ext = file_path.suffix.lower()
        if ext not in allowed_extensions:
            raise ValidationError(
                f"Invalid file extension '{ext}'. Allowed extensions: {allowed_extensions}"
            )
        return True

    @staticmethod
    def validate_csv_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validates whether DataFrame contains the required 52 features.
        Returns (is_valid, missing_columns).
        """
        cleaned_cols = [c.strip() for c in df.columns]
        missing = [f for f in FEATURE_COLUMNS if f not in cleaned_cols]
        return len(missing) == 0, missing

    @staticmethod
    def validate_feature_matrix(matrix: Any) -> bool:
        """Validates feature matrix shape and absence of invalid types."""
        if hasattr(matrix, "shape"):
            if len(matrix.shape) == 2 and matrix.shape[1] != len(FEATURE_COLUMNS):
                raise ValidationError(
                    f"Feature matrix column count mismatch: expected {len(FEATURE_COLUMNS)}, got {matrix.shape[1]}"
                )
            elif len(matrix.shape) == 3 and matrix.shape[2] != len(FEATURE_COLUMNS):
                raise ValidationError(
                    f"3D input feature dimension mismatch: expected {len(FEATURE_COLUMNS)}, got {matrix.shape[2]}"
                )
        return True
