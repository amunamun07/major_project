import yaml
import pytest
import pandas as pd

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)


@pytest.fixture()
def get_dataframe():
    return pd.read_csv(file_paths["dataset_path"])
