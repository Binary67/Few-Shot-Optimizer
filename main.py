import pandas as pd
from DataLoader import DataLoader
from ConfigManager import ConfigManager

if __name__ == "__main__":
    Config = ConfigManager("ConfigParams.yaml")

    SampleData = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [5, 4, 3, 2, 1],
        "Label": [0, 1, 0, 1, 0]
    })

    InputColumns = Config.GetParam("InputColumns")
    LabelColumn = Config.GetParam("LabelColumn")
    TestSize = Config.GetParam("TestSize")

    Loader = DataLoader(SampleData, InputColumns, LabelColumn)
    TrainX, TestX, TrainY, TestY = Loader.SplitDataset(TestSize=TestSize, RandomState=1)
    print("TrainX shape:", TrainX.shape)
    print("TestX shape:", TestX.shape)
    print("TrainY shape:", TrainY.shape)
    print("TestY shape:", TestY.shape)
