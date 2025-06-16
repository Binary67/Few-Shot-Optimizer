import os
import sys
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from LogManager import LogManager


def TestSetupCreatesLogFile(tmp_path):
    LogPath = tmp_path / "logs"
    Manager = LogManager(LogDirectory=str(LogPath), MaxLogs=5)
    FilePath = Manager.SetupLogging()
    assert os.path.exists(FilePath)


def TestOldLogsDeleted(tmp_path):
    LogPath = tmp_path / "logs"
    LogPath.mkdir()
    for Index in range(5):
        File = LogPath / f"20200101_00000{Index}.log"
        File.write_text("test")
        time.sleep(0.01)
    Manager = LogManager(LogDirectory=str(LogPath), MaxLogs=5)
    NewFile = Manager.SetupLogging()
    Files = sorted(os.listdir(LogPath))
    assert len(Files) == 5
    assert os.path.basename(NewFile) == Files[-1]
    assert "20200101_000000.log" not in Files


def TestLogFormatIncludesModuleAndFunction(tmp_path, caplog):
    LogPath = tmp_path / "logs"
    Manager = LogManager(LogDirectory=str(LogPath), MaxLogs=5)
    FilePath = Manager.SetupLogging()

    def DummyFunction():
        Logger = logging.getLogger(__name__)
        Logger.info("dummy message")

    DummyFunction()

    with open(FilePath, "r") as File:
        Lines = File.readlines()
    LastLine = Lines[-1] if Lines else ""
    assert __name__ in LastLine
    assert "DummyFunction" in LastLine

