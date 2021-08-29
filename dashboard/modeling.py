import matplotlib.pyplot as plt

import pandas as pd
import pickle

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def split_data(data, test_size=0.25, random_state=0):
    return train_test_split(data.drop(['label'], axis=1), data['label'], test_size=test_size, random_state=random_state)


class Clustering:
    def __init__(self):
        pass

    def k_means_clustering(self, data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        y_kmeans = kmeans.fit_predict(data)
        return y_kmeans

    def showclusters(self, data, n_clusters):
        label = data['label']
        y_kmeans = self.k_means_clustering(data.drop(['label'], axis=1), n_clusters)
        y_kmeans = pd.DataFrame(y_kmeans)
        new_df = pd.concat([y_kmeans, label], axis=1)
        new_df = new_df.rename(columns={0: 'cluster'})
        hard_clusters_list = []
        for i in range(n_clusters):
            count = new_df[new_df['cluster'] == i]['label'].value_counts()
            d = new_df.loc[new_df['label'].isin(count.index[count >= 50])]
            d = d['label'].value_counts()
            hard_clusters_list.append(d.index)
        hard_clusters_df = pd.DataFrame(hard_clusters_list)
        return hard_clusters_df.fillna('')


class ClassificationModels:
    def __init__(self, data):
        self.x_train, self.x_test, self.y_train, self.y_test = split_data(data)

    def logistic_regression(self):
        model = LogisticRegression()
        model.fit(self.x_train, self.y_train)
        return model

    def evaluate_model_performance(self, model):
        y_prediction = model.predict(self.x_test)
        plt. rcParams['figure.figsize'] = (10, 10)
        confuse_matrix = confusion_matrix(self.y_test, y_prediction)
        sns.heatmap(confuse_matrix, annot=True, cmap='Wistia')
        plt.title("Confusion Matrix for Logistic Regression")
        report = classification_report(self.y_test, y_prediction, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        return plt, df_report

    def predict_real_time_values(self, model, values):
        return model.predict(values)

    def save_the_model(self, model, model_path):
        pickle.dump(model, open(model_path, "wb"))













