import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_principle_components(data):
    scalar = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    principle_components = pca.fit_transform(scalar)
    principal_components_df = pd.DataFrame(
        data=principle_components,
        columns=["Principle component 1", "Principle component 2"],
    )
    return principal_components_df


def get_tsne_components(data):
    tsne = TSNE(n_components=2, learning_rate=50)
    tsne_components = tsne.fit_transform(data)
    tsne_components_df = pd.DataFrame(
        data=tsne_components,
        columns=["tsne component 1", "tsne component 2"],
    )
    return tsne_components_df
