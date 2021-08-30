import pickle
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def split_data(data, data_label, test_size=0.25, random_state=0):
    return train_test_split(
        data,
        data_label,
        test_size=test_size,
        random_state=random_state,
    )


class LRModel:
    def __init__(self, data, data_label):
        self.x_train, self.x_test, self.y_train, self.y_test = split_data(
            data, data_label
        )
        model = LogisticRegression()
        self.model = model.fit(self.x_train, self.y_train)

    def get_report(self):
        y_prediction = self.model.predict(self.x_test)
        # plt.rcParams["figure.figsize"] = (10, 10)
        # confuse_matrix = confusion_matrix(self.y_test, y_prediction)
        # sns.heatmap(confuse_matrix, annot=True, cmap="Wistia")
        # plt.title("Confusion Matrix for Logistic Regression")
        report = classification_report(self.y_test, y_prediction, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        return df_report

    def save_the_model(self, model_path):
        pickle.dump(self.model, open(model_path, "wb"))
