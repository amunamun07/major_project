from dashboard import Clustering
import pytest


@pytest.fixture()
def cluster_model(get_dataframe):
    return Clustering(
        get_dataframe.drop(["label"], axis=1), get_dataframe.label, n_clusters=4
    )


def test_get_silhouette_score(cluster_model):
    score = cluster_model.get_silhouette_score()
    assert (score < 1) and (score > -1)


def test_get_soft_clusters(cluster_model):
    calculated_df = cluster_model.get_soft_clusters()
    assert calculated_df.columns.values.tolist() == ["cluster", "label"]


def test_get_hard_clusters(cluster_model):
    calculated_df = cluster_model.get_hard_clusters()
    assert calculated_df.columns.tolist() == [0, 1, 2, 3]
