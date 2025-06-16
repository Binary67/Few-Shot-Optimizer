import pandas as pd
from DataLoader import DataLoader
from ConfigManager import ConfigManager

if __name__ == "__main__":
    Config = ConfigManager("ConfigParams.yaml")

    SampleData = pd.read_csv("SampleData.csv")

    InputColumns = Config.GetParam("InputColumns")
    LabelColumn = Config.GetParam("LabelColumn")
    TestSize = Config.GetParam("TestSize")

    Loader = DataLoader(SampleData, InputColumns, LabelColumn)
    TrainX, TestX, TrainY, TestY = Loader.SplitDataset(TestSize=TestSize, RandomState=1)
    print("TrainX shape:", TrainX.shape)
    print("TestX shape:", TestX.shape)
    print("TrainY shape:", TrainY.shape)
    print("TestY shape:", TestY.shape)
