from dashboard import get_principle_components
from dashboard import get_tsne_components


def test_get_principle_components(get_dataframe):
    calculated_value = get_principle_components(
        get_dataframe.drop(["label"], axis=1)
    ).shape
    assert (
        calculated_value[1] == 2
    ), "Check for reduction of 7 dimension features to 2 dimension with pca"


def test_get_tsne_components(get_dataframe):
    calculated_value = get_tsne_components(get_dataframe.drop(["label"], axis=1)).shape
    assert (
        calculated_value[1] == 2
    ), "Check for reduction of 7 dimension features to 2 dimension with tsne"
