import pandas as pd
import openai
from EvaluatePrompt import EvaluatePrompt


def Main() -> None:
    Data = [
        {"Text": "Hello", "Expected": "Hello"},
        {"Text": "Bye", "Expected": "Bye"},
    ]
    DataFrame = pd.DataFrame(Data)
    PromptTemplate = "Respond exactly with {Text}"

    try:
        Client = openai.AzureOpenAI()
    except Exception as Error:
        print(f"Azure client initialization failed: {Error}")
        return

    Evaluator = EvaluatePrompt(DataFrame, ["Text"], "Expected", PromptTemplate, Client)
    Accuracy, Result = Evaluator.RunEvaluation()
    print(f"Accuracy: {Accuracy}")
    print(Result)


if __name__ == "__main__":
    Main()
