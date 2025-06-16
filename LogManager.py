import os
import logging
from datetime import datetime
from glob import glob

class LogManager:
    def __init__(self, LogDirectory: str = "Log", MaxLogs: int = 5) -> None:
        self.LogDirectory = LogDirectory
        self.MaxLogs = MaxLogs
        os.makedirs(self.LogDirectory, exist_ok=True)

    def SetupLogging(self) -> str:
        for Handler in logging.root.handlers[:]:
            logging.root.removeHandler(Handler)
        FileName = datetime.now().strftime("%Y%m%d_%H%M%S.log")
        FilePath = os.path.join(self.LogDirectory, FileName)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
            handlers=[
                logging.FileHandler(FilePath),
                logging.StreamHandler()
            ]
        )
        self.DeleteOldLogs()
        return FilePath

    def DeleteOldLogs(self) -> None:
        LogFiles = sorted(glob(os.path.join(self.LogDirectory, "*.log")))
        while len(LogFiles) > self.MaxLogs:
            os.remove(LogFiles[0])
            LogFiles.pop(0)

