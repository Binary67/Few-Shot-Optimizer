import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, DataFrame: pd.DataFrame, TestSize: float = 0.2, ValidateSize: float = 0.2, RandomState: int = 42) -> None:
        self.DataFrame = DataFrame
        self.TestSize = TestSize
        self.ValidateSize = ValidateSize
        self.RandomState = RandomState

    def Split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        TempSize = self.TestSize + self.ValidateSize
        TrainData, TempData = train_test_split(
            self.DataFrame,
            test_size=TempSize,
            random_state=self.RandomState,
        )

        if TempData.empty:
            return TrainData, pd.DataFrame(), pd.DataFrame()

        TestRatio = self.TestSize / TempSize
        ValidateData, TestData = train_test_split(
            TempData,
            test_size=TestRatio,
            random_state=self.RandomState,
        )
        return TrainData, ValidateData, TestData
