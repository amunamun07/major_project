import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class Clustering:
    def __init__(self, data, data_label, n_clusters=4):
        self.data = data
        self.data_label = data_label
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            max_iter=300,
            n_init=10,
            random_state=0,
        )

    def get_soft_clusters(self):
        model_output = self.model.fit_predict(self.data)
        model_output = pd.DataFrame(model_output)
        df_including_cluster = pd.concat([model_output, self.data_label], axis=1)
        df_including_cluster = df_including_cluster.rename(columns={0: "cluster"})
        return df_including_cluster

    def get_hard_clusters(self):
        df_including_cluster = self.get_soft_clusters()
        hard_clusters_list = []
        for i in range(self.n_clusters):
            unique_label_counts = df_including_cluster[
                df_including_cluster["cluster"] == i
            ]["label"].value_counts()
            d = df_including_cluster.loc[
                df_including_cluster["label"].isin(
                    unique_label_counts.index[unique_label_counts >= 50]
                )
            ]
            d = d["label"].value_counts()
            hard_clusters_list.append(d.index)
        hard_clusters_df = pd.DataFrame(hard_clusters_list)
        hard_clusters_df = (hard_clusters_df.fillna("")).transpose()
        return hard_clusters_df

    def get_silhouette_score(self):
        score = silhouette_score(self.data, self.model.labels_, metric="euclidean")
        return score
