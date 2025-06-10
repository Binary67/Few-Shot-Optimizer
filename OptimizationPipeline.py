import os
import pandas as pd
import openai
from dotenv import load_dotenv
from EvaluatePrompt import EvaluatePrompt
from FewShotOptimizer import FewShotOptimizer
from SaveResults import SaveResults
from DataSplitter import DataSplitter

class OptimizationPipeline:
    def __init__(self,
                 DataFrame: pd.DataFrame,
                 FeatureColumns: list[str],
                 LabelColumns: list[str],
                 PromptTemplate: str,
                 MaxExamples: int = 5,
                 YamlPath: str = "OptimizedResults.yaml",
                 Client: openai.AzureOpenAI | None = None,
                 RandomState: int = 42) -> None:
        load_dotenv()
        self.DataFrame = DataFrame.copy()
        self.FeatureColumns = FeatureColumns
        self.LabelColumns = LabelColumns
        self.PromptTemplate = PromptTemplate
        self.MaxExamples = MaxExamples
        self.YamlPath = YamlPath
        self.RandomState = RandomState
        self.Client = Client or openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )

    def Run(self) -> dict:
        Splitter = DataSplitter(self.DataFrame, RandomState=self.RandomState)
        TrainData, ValidateData, TestData = Splitter.Split()

        Optimizer = FewShotOptimizer(
            TrainData=TrainData,
            ValidateData=ValidateData,
            FeatureColumns=self.FeatureColumns,
            LabelColumns=self.LabelColumns,
            BasePromptTemplate=self.PromptTemplate,
            MaxExamples=self.MaxExamples,
            Client=self.Client,
        )

        BestExamples, OptimizedPrompt, FinalAccuracy = Optimizer.OptimizeGreedy()

        BaselineEvaluator = EvaluatePrompt(
            ValidateData,
            self.FeatureColumns,
            self.LabelColumns,
            self.PromptTemplate,
            self.Client,
        )
        BaselineAccuracy, _ = BaselineEvaluator.RunEvaluation()

        TestEvaluator = EvaluatePrompt(
            TestData,
            self.FeatureColumns,
            self.LabelColumns,
            OptimizedPrompt,
            self.Client,
        )
        TestAccuracy, _ = TestEvaluator.RunEvaluation()

        BaselineTestEvaluator = EvaluatePrompt(
            TestData,
            self.FeatureColumns,
            self.LabelColumns,
            self.PromptTemplate,
            self.Client,
        )
        BaselineTestAccuracy, _ = BaselineTestEvaluator.RunEvaluation()

        ResultsSaver = SaveResults(self.YamlPath)
        ResultsSaver.SaveBestExamplesAndAccuracy(
            BestExamples=BestExamples,
            FinalAccuracy=FinalAccuracy,
            OptimizedPrompt=OptimizedPrompt,
            BaselineAccuracy=BaselineAccuracy,
            TestAccuracy=TestAccuracy,
            BaselineTestAccuracy=BaselineTestAccuracy,
            PromptTemplate=self.PromptTemplate,
        )

        return {
            "BestExamples": BestExamples,
            "OptimizedPrompt": OptimizedPrompt,
            "FinalAccuracy": FinalAccuracy,
            "BaselineAccuracy": BaselineAccuracy,
            "TestAccuracy": TestAccuracy,
            "BaselineTestAccuracy": BaselineTestAccuracy,
        }
