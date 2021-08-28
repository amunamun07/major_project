import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


class DataCleaning:
    def __init__(self):
        pass

    @staticmethod
    def convert_objects(df):
        le = LabelEncoder()
        df['department'] = le.fit_transform(df['department'])
        df['salary'] = le.fit_transform(df['salary'])
        return df

    @staticmethod
    def handle_data_imbalance(df, df_label):
        smote = SMOTE(k_neighbors=1)
        df, df_label = smote.fit_resample(df, df_label)
        return df, df_label

    def get_cleaned_data(self, data):
        data = self.convert_objects(data)
        return data
