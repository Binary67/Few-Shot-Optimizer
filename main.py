import os

import pandas as pd
import openai
from dotenv import load_dotenv
from EvaluatePrompt import EvaluatePrompt


def Main() -> None:
    load_dotenv()
    Data = [
        {"Text": "Hello", "Expected": "Hello"},
        {"Text": "Bye", "Expected": "Bye"},
    ]
    DataFrame = pd.DataFrame(Data)
    PromptTemplate = "Respond exactly with {Text}"

    Client = openai.AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )

    Evaluator = EvaluatePrompt(DataFrame, ["Text"], "Expected", PromptTemplate, Client)
    Accuracy, Result = Evaluator.RunEvaluation()
    print(f"Accuracy: {Accuracy}")
    print(Result)


if __name__ == "__main__":
    Main()
