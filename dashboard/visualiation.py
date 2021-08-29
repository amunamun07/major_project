import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

plt.style.use('seaborn')


class Visualization:
    def __init__(self):
        pass

    def distribution_plot(self, data):
        fig, axes = plt.subplots(3, 3)
        axes = axes.ravel()
        for col, ax in zip(data.columns, axes):
            sns.histplot(data=data[col], kde=True, stat='count', ax=ax)
        fig.tight_layout()
        return plt

    def elbow_plot(self, data):
        plt.rcParams['figure.figsize'] = (10, 4)
        within_cluster_sum_of_squares = []
        for index in range(1, 11):
            km = KMeans(n_clusters=index, init='k-means++', max_iter=300, n_init=10, random_state=0)
            km.fit(data)
            within_cluster_sum_of_squares.append(km.inertia_)
        plt.plot(range(1, 11), within_cluster_sum_of_squares)
        plt.title('Elbow Plot to find the number of Clusters', fontsize=20)
        plt.xlabel('No of clusters (K)')
        plt.ylabel('Within Cluster Sum of Squares (WCSS)')
        return plt




