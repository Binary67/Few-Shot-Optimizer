import os

import pandas as pd
import openai
from dotenv import load_dotenv

class EvaluatePrompt:
    def __init__(self, DataFrame: pd.DataFrame, FeatureColumns: list[str], LabelColumn: str, PromptTemplate: str, Client: openai.AzureOpenAI | None = None) -> None:
        load_dotenv()
        self.DataFrame = DataFrame.copy()
        self.FeatureColumns = FeatureColumns
        self.LabelColumn = LabelColumn
        self.PromptTemplate = PromptTemplate
        self.Client = Client or openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )

    def GeneratePredictions(self, Model: str | None = None, Temperature: float = 0.0) -> pd.DataFrame:
        ModelName = Model or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        Predictions = []
        for _, Row in self.DataFrame.iterrows():
            Prompt = self.PromptTemplate.format(**{Column: Row[Column] for Column in self.FeatureColumns})
            Response = self.Client.chat.completions.create(
                model=ModelName,
                messages=[{"role": "user", "content": Prompt}],
                temperature=Temperature,
            )
            Prediction = Response.choices[0].message.content.strip()
            Predictions.append(Prediction)
        self.DataFrame["ModelPrediction"] = Predictions
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
