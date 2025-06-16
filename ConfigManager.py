import yaml

class ConfigManager:
    def __init__(self, FilePath: str = "ConfigParams.yaml") -> None:
        self.FilePath = FilePath
        self.Params = self.LoadConfig()

    def LoadConfig(self) -> dict:
        with open(self.FilePath, 'r') as File:
            return yaml.safe_load(File)

    def GetParam(self, Key: str, Default=None):
        return self.Params.get(Key, Default)
