import yaml


def LoadPromptTemplate(FilePath: str) -> str:
    """Load the prompt template from a YAML file."""
    try:
        with open(FilePath, 'r', encoding='utf-8') as YamlFile:
            Data = yaml.safe_load(YamlFile) or {}
    except FileNotFoundError as Exc:
        raise FileNotFoundError(f"{FilePath} not found") from Exc
    except yaml.YAMLError as Exc:
        raise ValueError(f"Error parsing YAML file: {Exc}") from Exc

    if not Data.get("prompt_template"):
        raise ValueError("No prompt_template found in YAML file")
    return Data["prompt_template"]
