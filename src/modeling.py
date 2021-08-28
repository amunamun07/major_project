import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import streamlit as st
from sklearn.model_selection import train_test_split
from cleaning import DataCleaning
from feature_extraction import FeatureExtraction


def split_data(data, data_label, test_size=0.25, random_state=0):
    df, df_label = DataCleaning().handle_data_imbalance(data, data_label)
    return train_test_split(df, df_label, test_size=test_size, random_state=random_state)


class ModelBuilding:
    def __init__(self, data):
        self.data = data
        data = self.data.copy()
        self.target_data = data.iloc[:, 1:12]

    def plot_elbow(self):
        plt.rcParams['figure.figsize'] = (10, 4)
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            km.fit(self.target_data)
            wcss.append(km.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('The Elbow Method', fontsize=20)
        plt.xlabel('No of clusters')
        plt.ylabel('wcss')
        return plt

    # def k_means_clustering2(self):
    #     principle_components = FeatureExtraction(self.data[self.data['left'] == 1]).principle_component_analysis()
    #     km = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #     y_means = km.fit_predict(principle_components)
    #     print(y_means)
    #
    #     # # y_means = pd.DataFrame(y_means)
    #     # # z = pd.concat([y_means, self.data_label], axis=1)
    #     # # z = z.rename(columns={0: 'cluster'})
    #     #
    #     plt.scatter(self.data[y_means == 0, 0], self.data[y_means == 0, 1], s=100, c='pink', label='general')
    #     plt.scatter(self.data[y_means == 1, 0], self.data[y_means == 1, 1], s=100, c='yellow', label='spendthrift')
    #     plt.scatter(self.data[y_means == 2, 0], self.data[y_means == 2, 1], s=100, c='cyan', label='target')
    #     plt.title('Clustering between df and df_label', fontsize=20)
    #     return plt
    #     # st.write(plt)

    def k_means_clustering(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit_predict(self.target_data)
        clusters = pd.DataFrame(self.target_data, columns=self.data.drop("EmpID", axis=1).columns)
        clusters['label'] = kmeans.labels_
        polar = clusters.groupby("label").mean().reset_index()
        polar = pd.melt(polar, id_vars=["label"])
        fig = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True, height=800, width=1400)
        return fig









