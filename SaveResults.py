import os
import yaml
from typing import Any, Dict, List
from datetime import datetime


class SaveResults:
    def __init__(self, FilePath: str = "optimization_results.yaml") -> None:
        self.FilePath = FilePath

    def SaveBestExamplesAndAccuracy(self, BestExamples: List[Dict[str, Any]], FinalAccuracy: float, OptimizedPrompt: str = "", BaselineAccuracy: float = None, TestAccuracy: float = None, BaselineTestAccuracy: float = None, PromptTemplate: str = "") -> None:
        """Save the best examples and accuracy results to a YAML file."""

        ExistingData: Dict[str, Any] = {}
        if os.path.exists(self.FilePath):
            try:
                with open(self.FilePath, 'r', encoding='utf-8') as YamlFile:
                    ExistingData = yaml.safe_load(YamlFile) or {}
            except yaml.YAMLError as Error:
                print(f"Error parsing YAML file: {Error}")

        if "prompt_template" not in ExistingData and PromptTemplate:
            ExistingData["prompt_template"] = PromptTemplate

        ExistingData["optimization_results"] = {
            "timestamp": datetime.now().isoformat(),
            "validation_accuracy": float(FinalAccuracy),
            "validation_baseline_accuracy": float(BaselineAccuracy) if BaselineAccuracy is not None else None,
            "validation_improvement": float(FinalAccuracy - BaselineAccuracy) if BaselineAccuracy is not None else None,
            "test_accuracy": float(TestAccuracy) if TestAccuracy is not None else None,
            "test_baseline_accuracy": float(BaselineTestAccuracy) if BaselineTestAccuracy is not None else None,
            "test_improvement": float(TestAccuracy - BaselineTestAccuracy) if TestAccuracy is not None and BaselineTestAccuracy is not None else None,
            "optimized_prompt": OptimizedPrompt,
            "best_examples": [
                {
                    "example_id": i + 1,
                    "data": Example
                }
                for i, Example in enumerate(BestExamples)
            ],
            "total_examples": len(BestExamples)
        }

        with open(self.FilePath, 'w', encoding='utf-8') as YamlFile:
            yaml.dump(ExistingData, YamlFile, default_flow_style=False, allow_unicode=True, indent=2)
        
        print(f"Results saved to {self.FilePath}")

    def LoadResults(self) -> Dict[str, Any]:
        """Load results from the YAML file."""
        try:
            with open(self.FilePath, 'r', encoding='utf-8') as YamlFile:
                return yaml.safe_load(YamlFile)
        except FileNotFoundError:
            print(f"File {self.FilePath} not found.")
            return {}
        except yaml.YAMLError as Error:
            print(f"Error parsing YAML file: {Error}")
            return {}