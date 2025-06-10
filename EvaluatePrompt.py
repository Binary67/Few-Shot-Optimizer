import os
import re
import concurrent.futures
from typing import Tuple
import time
import random

import pandas as pd
import openai
from dotenv import load_dotenv

class EvaluatePrompt:
    def __init__(self, DataFrame: pd.DataFrame, FeatureColumns: list[str], LabelColumn: str, PromptTemplate: str, Client: openai.AzureOpenAI | None = None) -> None:
        load_dotenv()
        self.DataFrame = DataFrame.copy()
        self.FeatureColumns = FeatureColumns
        self.LabelColumn = LabelColumn
        self.PromptTemplate = self.SanitizePlaceholders(PromptTemplate)
        self.Client = Client or openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )

    @staticmethod
    def SanitizePlaceholders(Template: str) -> str:
        return re.sub(r"{['\"]([^'\"]+)['\"]}", r"{\1}", Template)

    def _GenerateSinglePrediction(self, Args: Tuple[int, pd.Series, str, float]) -> Tuple[int, str]:
        Index, Row, ModelName, Temperature = Args
        Prompt = self.PromptTemplate.format(**{Column: Row[Column] for Column in self.FeatureColumns})
        
        MaxRetries = 25
        BaseDelay = 1
        MaxDelay = 60
        
        for Attempt in range(MaxRetries):
            try:
                Response = self.Client.chat.completions.create(
                    model=ModelName,
                    messages=[{"role": "user", "content": Prompt}],
                    temperature=Temperature,
                )
                Prediction = Response.choices[0].message.content.strip()
                return Index, Prediction
            except Exception as Error:
                if Attempt == MaxRetries - 1:
                    raise Error
                
                Delay = min(BaseDelay * (2 ** Attempt) + random.uniform(0, 1), MaxDelay)
                time.sleep(Delay)
        
        raise Exception("Maximum retries exceeded")

    def GeneratePredictions(self, Model: str | None = None, Temperature: float = 0.0, MaxWorkers: int = 5) -> pd.DataFrame:
        ModelName = Model or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        
        TaskArgs = [(Index, Row, ModelName, Temperature) for Index, Row in self.DataFrame.iterrows()]
        
        Results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MaxWorkers) as Executor:
            FutureToIndex = {Executor.submit(self._GenerateSinglePrediction, Args): Args[0] for Args in TaskArgs}
            
            for Future in concurrent.futures.as_completed(FutureToIndex):
                Index, Prediction = Future.result()
                Results[Index] = Prediction
        
        PredictionsList = [Results[Index] for Index in self.DataFrame.index]
        self.DataFrame["ModelPrediction"] = PredictionsList
        return self.DataFrame

    def CalculateAccuracy(self) -> float:
        if "ModelPrediction" not in self.DataFrame.columns:
            raise ValueError("Predictions have not been generated.")
        Correct = (self.DataFrame["ModelPrediction"] == self.DataFrame[self.LabelColumn]).sum()
        return Correct / len(self.DataFrame)

    def RunEvaluation(self, Model: str | None = None, Temperature: float = 0.0):
        self.GeneratePredictions(Model=Model, Temperature=Temperature)
        Accuracy = self.CalculateAccuracy()
        return Accuracy, self.DataFrame
