import os
import pandas as pd
import openai
from dotenv import load_dotenv
from EvaluatePrompt import EvaluatePrompt

class FewShotOptimizer:
    def __init__(self, TrainData: pd.DataFrame, ValidateData: pd.DataFrame, FeatureColumns: list[str], 
                 LabelColumn: str, BasePromptTemplate: str, MaxExamples: int = 5, Client: openai.AzureOpenAI | None = None) -> None:
        load_dotenv()
        self.TrainData = TrainData.copy()
        self.ValidateData = ValidateData.copy()
        self.FeatureColumns = FeatureColumns
        self.LabelColumn = LabelColumn
        self.BasePromptTemplate = BasePromptTemplate
        self.MaxExamples = MaxExamples
        self.Client = Client or openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )
        self.BestExamples = []
        self.BaselineAccuracy = 0.0

    def CreateFewShotPrompt(self, Examples: list[dict]) -> str:
        if not Examples:
            return self.BasePromptTemplate
        
        ExampleTexts = []
        for Example in Examples:
            FeatureValues = {Column: Example[Column] for Column in self.FeatureColumns}
            ExampleInput = self.BasePromptTemplate.format(**FeatureValues)
            ExampleOutput = Example[self.LabelColumn]
            ExampleTexts.append(f"Input: {ExampleInput}\nOutput: {ExampleOutput}")
        
        FewShotSection = "\n\n".join(ExampleTexts)
        return f"Here are some examples:\n\n{FewShotSection}\n\nNow, please respond to:\n{self.BasePromptTemplate}"

    def EvaluateWithExamples(self, Examples: list[dict]) -> float:
        PromptTemplate = self.CreateFewShotPrompt(Examples)
        Evaluator = EvaluatePrompt(self.ValidateData, self.FeatureColumns, self.LabelColumn, PromptTemplate, self.Client)
        Accuracy, _ = Evaluator.RunEvaluation()
        return Accuracy

    def OptimizeGreedy(self) -> tuple[list[dict], str, float]:
        print("Starting greedy few-shot optimization...")
        
        self.BaselineAccuracy = self.EvaluateWithExamples([])
        print(f"Baseline accuracy (zero-shot): {self.BaselineAccuracy:.4f}")
        
        self.BestExamples = []
        BestAccuracy = self.BaselineAccuracy
        AvailableExamples = self.TrainData.to_dict('records')
        
        for Iteration in range(self.MaxExamples):
            print(f"\nIteration {Iteration + 1}/{self.MaxExamples}")
            
            BestCandidate = None
            BestCandidateAccuracy = BestAccuracy
            
            for Candidate in AvailableExamples:
                TestExamples = self.BestExamples + [Candidate]
                TestAccuracy = self.EvaluateWithExamples(TestExamples)
                
                if TestAccuracy > BestCandidateAccuracy:
                    BestCandidate = Candidate
                    BestCandidateAccuracy = TestAccuracy
            
            if BestCandidate is None:
                print(f"No improvement found. Stopping at {len(self.BestExamples)} examples.")
                break
            
            self.BestExamples.append(BestCandidate)
            AvailableExamples.remove(BestCandidate)
            BestAccuracy = BestCandidateAccuracy
            
            print(f"Added example: {BestCandidate}")
            print(f"New accuracy: {BestAccuracy:.4f} (improvement: {BestAccuracy - self.BaselineAccuracy:.4f})")
        
        FinalPrompt = self.CreateFewShotPrompt(self.BestExamples)
        
        print(f"\nOptimization complete!")
        print(f"Best examples selected: {len(self.BestExamples)}")
        print(f"Final accuracy: {BestAccuracy:.4f}")
        print(f"Total improvement: {BestAccuracy - self.BaselineAccuracy:.4f}")
        
        return self.BestExamples, FinalPrompt, BestAccuracy

    def GetOptimizedPrompt(self) -> str:
        if not self.BestExamples:
            raise ValueError("Optimization has not been run. Call OptimizeGreedy() first.")
        return self.CreateFewShotPrompt(self.BestExamples)

    def GetResults(self) -> dict:
        return {
            "BestExamples": self.BestExamples,
            "BaselineAccuracy": self.BaselineAccuracy,
            "OptimizedPrompt": self.GetOptimizedPrompt() if self.BestExamples else self.BasePromptTemplate,
            "NumberOfExamples": len(self.BestExamples)
        }