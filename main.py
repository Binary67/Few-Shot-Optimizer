import pandas as pd
from dotenv import load_dotenv
from OptimizationPipeline import OptimizationPipeline


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

    # Load or create your dataframe here
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
    ]
    DataFrame = pd.DataFrame(Data)

    Pipeline = OptimizationPipeline(
        DataFrame=DataFrame,
        FeatureColumns=FeatureColumns,
        LabelColumns=LabelColumns,
        PromptTemplate=PromptTemplate,
        MaxExamples=5,
        YamlPath=YamlPath,
    )

    Pipeline.Run()

if __name__ == "__main__":
    Main()
