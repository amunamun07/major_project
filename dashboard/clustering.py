from sklearn.cluster import KMeans
import pandas as pd


class Clustering:
    def __init__(self, data, data_label):
        self.data = data
        self.data_label = data_label

    @staticmethod
    def kmeans_clustering(data, n_clusters):
        kmeans_model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            max_iter=300,
            n_init=10,
            random_state=0,
        )
        kmeans_output = kmeans_model.fit_predict(data)
        return kmeans_output

    def show_clusters(self, n_clusters):
        kmeans_output = self.kmeans_clustering(self.data, n_clusters)
        kmeans_output = pd.DataFrame(kmeans_output)
        df_including_cluster = pd.concat([kmeans_output, self.data_label], axis=1)
        df_including_cluster = df_including_cluster.rename(columns={0: "cluster"})
        hard_clusters_list = []
        for i in range(n_clusters):
            unique_label_counts = df_including_cluster[df_including_cluster["cluster"] == i]["label"].value_counts()
            d = df_including_cluster.loc[
                                        df_including_cluster["label"].isin(unique_label_counts.
                                                                           index[unique_label_counts >= 50])
                                        ]
            d = d["label"].value_counts()
            hard_clusters_list.append(d.index)
        hard_clusters_df = pd.DataFrame(hard_clusters_list)
        return (hard_clusters_df.fillna("")).transpose()
