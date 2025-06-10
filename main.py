import pandas as pd
from dotenv import load_dotenv
from OptimizationPipeline import OptimizationPipeline


def Main() -> None:
    load_dotenv()

    YamlPath = "OptimizedResults.yaml"
    PromptTemplate = 'Transform this greeting: {Text}'

    # Define feature columns and label column for your dataset
    FeatureColumns = ["Text"]
    LabelColumn = "Expected"

    # Load or create your dataframe here
    Data = [
        {"Text": "Hello", "Expected": "Goodbye"},
        {"Text": "Bye", "Expected": "Bye"},
        {"Text": "Good morning", "Expected": "Goodbye"},
        {"Text": "Thank you", "Expected": "Goodbye"},
        {"Text": "Please", "Expected": "Please"},
    ]
    DataFrame = pd.DataFrame(Data)

    Pipeline = OptimizationPipeline(
        DataFrame=DataFrame,
        FeatureColumns=FeatureColumns,
        LabelColumn=LabelColumn,
        PromptTemplate=PromptTemplate,
        MaxExamples=5,
        YamlPath=YamlPath,
    )

    Pipeline.Run()

if __name__ == "__main__":
    Main()
