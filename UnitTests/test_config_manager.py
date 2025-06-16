import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ConfigManager import ConfigManager


def TestConfigManagerLoadsParams():
    Manager = ConfigManager("ConfigParams.yaml")
    assert Manager.GetParam("TestSize") == 0.4
    assert Manager.GetParam("InputColumns") == ["A", "B"]
    assert Manager.GetParam("LabelColumn") == "Label"


def TestConfigManagerDefault():
    Manager = ConfigManager("ConfigParams.yaml")
    assert Manager.GetParam("NonExisting", "Default") == "Default"
