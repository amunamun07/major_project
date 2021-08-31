from dashboard import DataInformation


def test_get_data_information(get_dataframe):
    calculated_shape = DataInformation(get_dataframe).get_shape()
    expected_shape = (2200, 8)
    assert (
        calculated_shape == expected_shape
    ), "Shape might change if more data is added in dataset"


def test_get_no_of_null(get_dataframe):
    calculated_value = DataInformation(get_dataframe).get_no_of_null()
    expected_value = "0 null values found !"
    assert (
        calculated_value == expected_value
    ), " Test Pass if there is no Missing Values"


def test_get_categorical_data_columns(get_dataframe):
    calculated_type = DataInformation(get_dataframe).get_categorical_data_columns()
    expected_type = object
    assert calculated_type.dtype == expected_type, "Categorical datatype must be object"


def test_get_data_statistics(get_dataframe):
    calculated_statistics = DataInformation(get_dataframe).get_data_statistics(
        selected_target="Overall"
    )
    expected_statistics = get_dataframe.describe()
    breakpoint()
    assert type(calculated_statistics) == type(expected_statistics), (
        "Selected target is the name of unique " "labels and also the overall data "
    )
