import numpy as np
from sklearn.decomposition import PCA
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


class FeatureExtraction:
    def __init__(self, data):
        self.data = data

    def principle_component_analysis(self):
        scaler = StandardScaler()
        scalar_pca1 = scaler.fit_transform(self.data[['satisfaction_level']])
        scalar_pca2 = scaler.fit_transform(self.data[['number_project',
                                                      'last_evaluation',
                                                      'promotion_last_5years']])
        pca = PCA(n_components=1)
        principle_component_1 = pca.fit_transform(scalar_pca1)
        principle_component_2 = pca.fit_transform(scalar_pca2)
        principle_component = np.concatenate((principle_component_1,
                                              principle_component_2
                                              ),
                                             axis=1)
        print(principle_component_2.shape)
        principal_components_df = pd.DataFrame(
            data=principle_component,
            columns=["Principle component 1", "Principle component 2"],
        )

        return principal_components_df

    def visualize_pca(self, targets, colors):
        """Visualing the pricinple components of images on 2D scatter plot."""
        plt.figure(figsize=(10, 8))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Principal Component - 1", fontsize=17)
        plt.ylabel("Principal Component - 2", fontsize=17)
        plt.title("Principal Component Analysis of HR Dataset", fontsize=18, pad=15)
        for target, color in zip(targets, colors):
            indices_to_keep = self.data['left'] == target
            print(self.data['left'])
            plt.scatter(
                self.principle_component_analysis().loc[
                    indices_to_keep, "Principle component 2"
                ],
                self.principle_component_analysis().loc[
                    indices_to_keep, "Principle component 1"
                ],
                c=color,
                s=40,
            )

        plt.legend(targets, prop={"size": 15})
        st.pyplot(plt)

    def tsne_analysis(self):
        tsne = TSNE(learning_rate=50)
        tsne_components = tsne.fit_transform(self.data)
        self.data["x"] = tsne_components[:, 0]
        self.data["y"] = tsne_components[:, 1]
        sns.scatterplot(x="x", y="y", data=self.data)
        st.write(plt)



