import pandas as pd
import logging
from DataLoader import DataLoader
from ConfigManager import ConfigManager
from LogManager import LogManager

if __name__ == "__main__":
    Config = ConfigManager("ConfigParams.yaml")

    LogMgr = LogManager()
    LogFile = LogMgr.SetupLogging()
    Logger = logging.getLogger(__name__)
    Logger.info(f"Log file created at {LogFile}")

    SampleData = pd.read_csv("SampleData.csv")

    InputColumns = Config.GetParam("InputColumns")
    LabelColumn = Config.GetParam("LabelColumn")
    TestSize = Config.GetParam("TestSize")

    Loader = DataLoader(SampleData, InputColumns, LabelColumn)
    TrainX, TestX, TrainY, TestY = Loader.SplitDataset(TestSize=TestSize, RandomState=1)
    Logger.info("TrainX shape: %s", TrainX.shape)
    Logger.info("TestX shape: %s", TestX.shape)
    Logger.info("TrainY shape: %s", TrainY.shape)
    Logger.info("TestY shape: %s", TestY.shape)

