import pandas as pd
from dotenv import load_dotenv
from DataSplitter import DataSplitter
from FewShotOptimizer import FewShotOptimizer
from EvaluatePrompt import EvaluatePrompt
from SaveResults import SaveResults


def Main() -> None:
    load_dotenv()

    YamlPath = "OptimizedResults.yaml"
    PromptTemplate = (
        'Determine if the following text expresses a career aspiration. '
        'Respond with JSON containing "IsAspiration" and "Substring". Text: {Text}'
    )

    # Define feature columns and label columns for your dataset
    FeatureColumns = ["Text"]
    LabelColumns = ["IsAspiration", "Substring"]

    # Load or create your dataframe here with additional examples
    Data = [
        {
            "Text": "I aspire to become a data scientist one day",
            "IsAspiration": "has_aspiration",
            "Substring": "become a data scientist"
        },
        {
            "Text": "I enjoy hiking on weekends",
            "IsAspiration": "no_aspiration",
            "Substring": ""
        },
        {
            "Text": "My dream is to lead a large engineering team",
            "IsAspiration": "has_aspiration",
            "Substring": "lead a large engineering team"
        },
        {
            "Text": "Thank you for your time",
            "IsAspiration": "no_aspiration",
            "Substring": ""
        },
        {
            "Text": "One day I hope to start my own company",
            "IsAspiration": "has_aspiration",
            "Substring": "start my own company"
        },
        {
            "Text": "I'm planning to enroll in a graduate program next year",
            "IsAspiration": "has_aspiration",
            "Substring": "enroll in a graduate program"
        },
        {
            "Text": "Coffee is my favorite beverage",
            "IsAspiration": "no_aspiration",
            "Substring": ""
        },
        {
            "Text": "My goal is to become a senior software engineer",
            "IsAspiration": "has_aspiration",
            "Substring": "become a senior software engineer"
        },
        {
            "Text": "The weather is nice today",
            "IsAspiration": "no_aspiration",
            "Substring": ""
        },
        {
            "Text": "I hope to write a novel someday",
            "IsAspiration": "has_aspiration",
            "Substring": "write a novel"
        }
    ]
    DataFrame = pd.DataFrame(Data)

    Splitter = DataSplitter(DataFrame)
    TrainData, ValidateData, _ = Splitter.Split()

    Optimizer = FewShotOptimizer(
        TrainData=TrainData,
        ValidateData=ValidateData,
        FeatureColumns=FeatureColumns,
        LabelColumns=LabelColumns,
        BasePromptTemplate=PromptTemplate,
        MaxExamples=5,
    )

    Optimizer.OptimizeGreedy()

    OptimizedPrompt = Optimizer.GetOptimizedPrompt()

    Validator = EvaluatePrompt(
        ValidateData,
        FeatureColumns,
        LabelColumns,
        OptimizedPrompt,
    )
    Accuracy, _ = Validator.RunEvaluation()

    ResultsSaver = SaveResults(YamlPath)
    ResultsSaver.SaveBestExamplesAndAccuracy(
        BestExamples=Optimizer.BestExamples,
        FinalAccuracy=Accuracy,
        OptimizedPrompt=OptimizedPrompt,
        BaselineAccuracy=Optimizer.BaselineAccuracy,
        PromptTemplate=PromptTemplate,
    )

    print(f"Validation accuracy with optimized prompt: {Accuracy:.4f}")

if __name__ == "__main__":
    Main()
