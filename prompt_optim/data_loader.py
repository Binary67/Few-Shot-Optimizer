import ast
import pandas as pd
from typing import Tuple, List, Optional


class DataLoader:
    """Utility class for ingesting and validating training data."""

    REQUIRED_COLUMNS = ["Input", "Label"]

    @staticmethod
    def load_dataframe(file_path: str) -> pd.DataFrame:
        """Load CSV file into DataFrame and validate columns."""
        df = pd.read_csv(file_path)
        if list(df.columns) != DataLoader.REQUIRED_COLUMNS:
            raise ValueError(
                f"DataFrame must have columns {DataLoader.REQUIRED_COLUMNS}, found {list(df.columns)}"
            )
        return df

    @staticmethod
    def validate_types(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure Input column is str and normalize Label column to List[str]."""
        df["Input"] = df["Input"].astype(str)
        NormalizedLabels: List[List[str]] = []
        for LabelValue in df["Label"]:
            if isinstance(LabelValue, str):
                Stripped = LabelValue.strip()
                try:
                    Evaluated = ast.literal_eval(Stripped)
                    if isinstance(Evaluated, list):
                        NormalizedLabels.append([str(Item) for Item in Evaluated])
                        continue
                except Exception:
                    pass
                NormalizedLabels.append([Stripped])
            elif isinstance(LabelValue, list):
                NormalizedLabels.append([str(Item) for Item in LabelValue])
            else:
                raise TypeError("Label column must contain strings or list of strings")
        df["Label"] = NormalizedLabels
        return df

    @staticmethod
    def train_dev_split(
        df: pd.DataFrame, dev_ratio: Optional[float] = None, random_state: int = 42
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Optionally split into train and dev sets."""
        if dev_ratio is None or dev_ratio <= 0 or dev_ratio >= 1:
            return df, None
        dev_df = df.sample(frac=dev_ratio, random_state=random_state)
        train_df = df.drop(dev_df.index)
        return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)

    @staticmethod
    def ingest(file_path: str, dev_ratio: Optional[float] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        df = DataLoader.load_dataframe(file_path)
        df = DataLoader.validate_types(df)
        return DataLoader.train_dev_split(df, dev_ratio)
