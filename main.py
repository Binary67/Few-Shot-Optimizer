from prompt_optim.data_loader import DataLoader


def main(DatasetPath: str = "UnitTests/assets/sample.csv", DevRatio: float | None = None) -> None:
    TrainDf, DevDf = DataLoader.ingest(DatasetPath, DevRatio)

    print("Loaded", len(TrainDf), "training rows")
    if DevDf is not None:
        print("Loaded", len(DevDf), "dev rows")


if __name__ == "__main__":
    main()
