import pandas as pd
from DataLoader import DataLoader

if __name__ == "__main__":
    SampleData = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [5, 4, 3, 2, 1],
        "Label": [0, 1, 0, 1, 0]
    })

    Loader = DataLoader(SampleData, ["A", "B"], "Label")
    TrainX, TestX, TrainY, TestY = Loader.SplitDataset(TestSize=0.4, RandomState=1)
    print("TrainX shape:", TrainX.shape)
    print("TestX shape:", TestX.shape)
    print("TrainY shape:", TrainY.shape)
    print("TestY shape:", TestY.shape)
