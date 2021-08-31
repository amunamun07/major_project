import pickle
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def split_data(data, data_label, test_size=0.25, random_state=0):
    """Split the data into train and validation set.

    Args:
        data (dataframe): With the numerical columns except label
        data_label (dataframe): With the label column
        test_size (float): Ratio of test set to be split from the data
        random_state (int): used to seed a new RandomState object.

    """
    return train_test_split(
        data,
        data_label,
        test_size=test_size,
        random_state=random_state,
    )


class LRModel:
    def __init__(self, data, data_label):
        """Fits a LR model to the data and returns classification report.

        Args:
            data (dataframe): With the numerical columns except label
            data_label (dataframe): With the label column

        """
        self.x_train, self.x_test, self.y_train, self.y_test = split_data(
            data, data_label
        )
        model = LogisticRegression()
        self.model = model.fit(self.x_train, self.y_train)

    def get_report(self):
        """Returns the classification report"""
        y_prediction = self.model.predict(self.x_test)
        report = classification_report(self.y_test, y_prediction, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        return df_report

    def save_the_model(self, model_path):
        """Saves the model.

        Args:
            model_path (str): Path of the model

        """
        pickle.dump(self.model, open(model_path, "wb"))
