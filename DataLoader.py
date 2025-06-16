import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List

class DataLoader:
    def __init__(self, DataFrame: pd.DataFrame, InputColumns, LabelColumn: str):
        self.DataFrame = DataFrame
        if isinstance(InputColumns, list):
            self.InputColumns = InputColumns
        else:
            self.InputColumns = [InputColumns]
        self.LabelColumn = LabelColumn
        self.ValidateSchema()

    def ValidateSchema(self) -> bool:
        MissingColumns = [Col for Col in self.InputColumns + [self.LabelColumn] if Col not in self.DataFrame.columns]
        if MissingColumns:
            raise ValueError(f"The following columns are missing: {MissingColumns}")
        return True

    def SplitDataset(self, TestSize: float = 0.2, RandomState: int = 42, Shuffle: bool = True):
        self.ValidateSchema()
        X = self.DataFrame[self.InputColumns]
        Y = self.DataFrame[self.LabelColumn]
        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=TestSize, random_state=RandomState, shuffle=Shuffle)
        return XTrain, XTest, YTrain, YTest
