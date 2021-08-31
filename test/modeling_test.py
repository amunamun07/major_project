from dashboard import LRModel


def test_get_report(get_dataframe):
    report = LRModel(
        get_dataframe.drop(["label"], axis=1), get_dataframe.label
    ).get_report()
    assert report.columns.values.tolist() == [
        "precision",
        "recall",
        "f1-score",
        "support",
    ]
