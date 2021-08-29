import numpy as np


class DataInformation:
    def __init__(self, df):
        self.df = df

    def get_dataset_head(self):
        return self.df.head(12)

    def get_shape(self):
        return self.df.shape

    def get_dataset_columns(self):
        return self.df.columns

    def get_all_data(self):
        return self.df

    def get_data_label(self):
        return self.df.label

    def get_data_excluding_label(self, only_include=None):
        if only_include is not None:
            return only_include
        else:
            return self.df.drop(['label'], axis=1)

    def get_no_of_null(self):
        return f'{self.df.isnull().sum().sum()} null values found !'

    def get_categorical_data_columns(self):
        object_data = self.df.select_dtypes("object")
        if not object_data.empty:
            return object_data.columns
        else:
            return "No Categorical Data Found !"

    def get_selected_column(self, selected_column):
        return self.df[selected_column]

    def check_data_imbalance(self):
        return self.df["label"].value_counts()

    def get_data_statistics(self, selected_target):
        if selected_target == "Overall":
            return self.df.describe()
        else:
            return self.df[self.df.label == selected_target].describe()

    def get_unique_labels(self, with_overall=False):
        if with_overall:
            return np.insert(self.df.label.unique(), 0, "Overall")
        else:
            return self.df.label.unique()


