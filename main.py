import os

import pandas as pd
import openai
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from EvaluatePrompt import EvaluatePrompt
from FewShotOptimizer import FewShotOptimizer
from SaveResults import SaveResults


def Main() -> None:
    load_dotenv()
    YamlPath = "OptimizedResults.yaml"
    PromptTemplate = 'Transform this greeting: {Text}'

    # Configuration - modify these to change feature columns and label
    FeatureColumns = ["Text"]
    LabelColumn = "Expected"

    Data = [
        {"Text": "Hello", "Expected": "Goodbye"},
        {"Text": "Bye", "Expected": "Bye"},
        {"Text": "Good morning", "Expected": "Goodbye"},
        {"Text": "Thank you", "Expected": "Goodbye"},
        {"Text": "Please", "Expected": "Please"},
        {"Text": "Excuse me", "Expected": "Excuse me"},
        {"Text": "Sorry", "Expected": "Goodbye"},
        {"Text": "Welcome", "Expected": "Welcome"},
        {"Text": "How are you", "Expected": "How are you"},
        {"Text": "See you later", "Expected": "See you later"},
        {"Text": "Nice to meet you", "Expected": "Nice to meet you"},
        {"Text": "Have a great day", "Expected": "Have a great day"},
        {"Text": "Good luck", "Expected": "Good luck"},
        {"Text": "Congratulations", "Expected": "Goodbye"},
        {"Text": "Happy birthday", "Expected": "Happy birthday"},
        {"Text": "Get well soon", "Expected": "Get well soon"},
        {"Text": "Take care", "Expected": "Goodbye"},
        {"Text": "Safe travels", "Expected": "Safe travels"},
        {"Text": "Good night", "Expected": "Good night"},
        {"Text": "Sweet dreams", "Expected": "Goodbye"},
    ]
    DataFrame = pd.DataFrame(Data)

    TrainData, TempData = train_test_split(DataFrame, test_size=0.5, random_state=42)
    ValidateData, TestData = train_test_split(TempData, test_size=0.5, random_state=42)

    Client = openai.AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )

    print("\n--- Few-Shot Optimization ---")
    MaxExamples = 5
    Optimizer = FewShotOptimizer(
        TrainData=TrainData,
        ValidateData=ValidateData,
        FeatureColumns=FeatureColumns,
        LabelColumn=LabelColumn,
        BasePromptTemplate=PromptTemplate,
        MaxExamples=MaxExamples,
        Client=Client
    )

    BestExamples, OptimizedPrompt, FinalAccuracy = Optimizer.OptimizeGreedy()
    print(f"Examples used: {len(BestExamples)} / {MaxExamples}")
    
    print(f"\nFinal Accuracy: {FinalAccuracy:.4f}")
    
    print("\n--- Baseline Comparison ---")
    BaselineEvaluator = EvaluatePrompt(ValidateData, FeatureColumns, LabelColumn, PromptTemplate, Client)
    BaselineAccuracy, _ = BaselineEvaluator.RunEvaluation()
    print(f"Baseline (zero-shot): {BaselineAccuracy:.4f}")
    print(f"Optimized (few-shot): {FinalAccuracy:.4f}")
    print(f"Improvement: {FinalAccuracy - BaselineAccuracy:.4f}")
    
    print("\n--- Test Set Evaluation ---")
    TestEvaluator = EvaluatePrompt(TestData, FeatureColumns, LabelColumn, OptimizedPrompt, Client)
    TestAccuracy, _ = TestEvaluator.RunEvaluation()
    print(f"Test Set Accuracy (with optimized prompt): {TestAccuracy:.4f}")
    
    BaselineTestEvaluator = EvaluatePrompt(TestData, FeatureColumns, LabelColumn, PromptTemplate, Client)
    BaselineTestAccuracy, _ = BaselineTestEvaluator.RunEvaluation()
    print(f"Test Set Baseline Accuracy: {BaselineTestAccuracy:.4f}")
    print(f"Test Set Improvement: {TestAccuracy - BaselineTestAccuracy:.4f}")
    
    # Save results to YAML file
    ResultsSaver = SaveResults(YamlPath)
    ResultsSaver.SaveBestExamplesAndAccuracy(
        BestExamples=BestExamples,
        FinalAccuracy=FinalAccuracy,
        OptimizedPrompt=OptimizedPrompt,
        BaselineAccuracy=BaselineAccuracy,
        TestAccuracy=TestAccuracy,
        BaselineTestAccuracy=BaselineTestAccuracy,
        PromptTemplate=PromptTemplate
    )

if __name__ == "__main__":
    Main()
