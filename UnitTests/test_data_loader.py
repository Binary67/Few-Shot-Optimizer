import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from DataLoader import DataLoader

def TestValidateSchemaPasses():
    Frame = pd.DataFrame({"talent_statement": [1], "leadership_attribute": [2], "Label": [0]})
    Loader = DataLoader(Frame, ["talent_statement", "leadership_attribute"], "Label")
    assert Loader.ValidateSchema()


def TestValidateSchemaFails():
    Frame = pd.DataFrame({"talent_statement": [1], "Label": [0]})
    with pytest.raises(ValueError):
        DataLoader(Frame, ["talent_statement", "leadership_attribute"], "Label")


def TestSplitDatasetShapes():
    Frame = pd.DataFrame({
        "talent_statement": range(10),
        "leadership_attribute": range(10),
        "Label": [0,1]*5
    })
    Loader = DataLoader(Frame, ["talent_statement", "leadership_attribute"], "Label")
    TrainX, TestX, TrainY, TestY = Loader.SplitDataset(TestSize=0.3, RandomState=42)
    assert len(TrainX) == 7
    assert len(TestX) == 3
    assert len(TrainY) == 7
    assert len(TestY) == 3
