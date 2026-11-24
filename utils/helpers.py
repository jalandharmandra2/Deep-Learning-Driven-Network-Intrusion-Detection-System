import json
from datetime import datetime
from typing import Any, Dict

def format_timestamp(ts: float = None) -> str:
    """Formats epoch timestamp or current time into ISO-like string."""
    dt = datetime.fromtimestamp(ts) if ts else datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def save_json(data: Dict[str, Any], file_path: Any) -> None:
    """Safely saves dictionary to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_json(file_path: Any) -> Dict[str, Any]:
    """Loads dictionary from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_column_names(df_columns: list) -> list:
    """Strips leading/trailing whitespaces and special characters from column names."""
    return [str(col).strip() for col in df_columns]
