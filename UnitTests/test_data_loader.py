from prompt_optim.data_loader import DataLoader


def test_ingest_and_split(tmp_path):
    CsvPath = tmp_path / "data.csv"
    CsvPath.write_text('Input,Label\nfoo,bar\nbar,"[\'baz\',\'qux\']"\n')
    TrainDf, DevDf = DataLoader.ingest(str(CsvPath), dev_ratio=0.5)
    assert list(TrainDf.columns) == ["Input", "Label"]
    assert isinstance(TrainDf["Input"].iloc[0], str)
    assert isinstance(TrainDf["Label"].iloc[0], list)
    assert TrainDf.shape[0] + (DevDf.shape[0] if DevDf is not None else 0) == 2


def test_label_list_string_parsed(tmp_path):
    CsvPath = tmp_path / "list.csv"
    CsvPath.write_text('Input,Label\nfoo,"[\'x\',\'y\']"\n')
    DataFrame, _ = DataLoader.ingest(str(CsvPath))
    assert DataFrame["Label"].iloc[0] == ["x", "y"]
