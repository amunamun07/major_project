import yaml
import pandas as pd
import streamlit as st

from modeling import LRModel
from clustering import Clustering
from data_info import DataInformation
from visualization import Visualization
from feature_extraction import get_principle_components, get_tsne_components

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)

st.set_option("deprecation.showPyplotGlobalUse", False)
st.title("Crop Production Dashboard")


def display_user_interface():
    data = st.file_uploader(label="", type=["csv", "xlsx"])
    if data is not None:
        df = pd.read_csv(data)
        option = st.sidebar.selectbox("Select Option", file_paths["drop_down_list"])

        data_info = DataInformation(df)
        visuals = Visualization()
        cluster = Clustering(
            data_info.get_data_excluding_label(),
            data_info.get_data_label(),
            n_clusters=4,
        )
        if option == "Data Analysis":
            st.subheader("Understanding the Data")
            if st.checkbox("Show Dataset"):
                st.dataframe(data_info.get_dataset_head())

            if st.checkbox("Display Shape"):
                st.write(data_info.get_shape())

            if st.checkbox("Check Null"):
                st.write(data_info.get_no_of_null())

            if st.checkbox("Check Individual Column"):
                selected_column = st.multiselect("Select Desired Column", df.columns)
                if selected_column:
                    st.dataframe(data_info.get_selected_column(selected_column))

            if st.checkbox("Check object data types"):
                st.write(data_info.get_categorical_data_columns())

            if st.checkbox("Check for data imbalance"):
                st.write(data_info.check_data_imbalance())

        elif option == "EDA":
            st.subheader("Exploratory Data Analysis")

            st.subheader("1. Data Exploration")
            if st.checkbox("High Level Statistics"):
                dropdown_list = data_info.get_unique_labels(with_overall=True)
                selected_target = st.selectbox("Select Target", dropdown_list)
                st.write(data_info.get_data_statistics(selected_target))

            st.subheader("2. Visualizations")
            st.subheader("2.1. All Data")
            if st.checkbox("Distribution Plot"):
                selected_column = st.multiselect(
                    "Select the columns you want to visualize",
                    data_info.get_dataset_columns(),
                )
                if selected_column:
                    returned_data = data_info.get_selected_column(selected_column)
                    st.dataframe(returned_data)
                    visuals.plot_data_distribution(
                        data_info.get_data_excluding_label(only_include=returned_data)
                    )
                else:
                    visuals.plot_data_distribution(data_info.get_data_excluding_label())
                st.pyplot()

            if st.checkbox("Show Correlation Heatmap"):
                visuals.plot_correlation_heatmap(data_info.get_data_excluding_label())
                st.pyplot()

            if st.checkbox("Find Optimal Number of Clusters for Kmeans"):
                visuals.plot_elbow_graph(data_info.get_data_excluding_label())
                st.pyplot()

            st.subheader("2.2. With Principle Components")
            if st.checkbox("Find Optimal Number of Clusters for Kmeans after PCA"):
                visuals.plot_elbow_graph(
                    get_principle_components(data_info.get_data_excluding_label())
                )
                st.pyplot()

            if st.checkbox("Latent Representation of Clusters after PCA"):
                visuals.plot_pca_scatter(
                    get_principle_components(data_info.get_data_excluding_label()),
                    cluster.get_soft_clusters().cluster,
                    cluster.get_hard_clusters().columns.tolist(),
                    file_paths["colors"],
                )
                st.pyplot()

            st.subheader("2.3. With TSNE Components")
            if st.checkbox("Find Optimal Number of Clusters for Kmeans after TSNE"):
                visuals.plot_elbow_graph(
                    get_tsne_components(data_info.get_data_excluding_label())
                )
                st.pyplot()

            if st.checkbox("Latent Representation of Clusters after TSNE"):
                visuals.plot_tsne_scatter(
                    get_tsne_components(data_info.get_data_excluding_label()),
                    cluster.get_soft_clusters().cluster,
                    cluster.get_hard_clusters().columns.tolist(),
                    file_paths["colors"],
                )
                st.pyplot()

        elif option == "Cluster Analysis":
            st.subheader("Clustering")
            if st.checkbox("K-means clustering"):
                df_clusters = cluster.get_hard_clusters()
                st.dataframe(df_clusters)
                st.write(f"Evaluation score is {cluster.get_silhouette_score()}")

            if st.checkbox("K-means clustering with Principle Components"):
                pca_cluster = Clustering(
                    get_principle_components(data_info.get_data_excluding_label()),
                    data_info.get_data_label(),
                    n_clusters=4,
                )
                df_clusters = pca_cluster.get_hard_clusters()
                st.dataframe(df_clusters)
                st.write(f"Evaluation score is {pca_cluster.get_silhouette_score()}")

            if st.checkbox("K-means clustering with TSNE Components"):
                tsne_cluster = Clustering(
                    get_tsne_components(data_info.get_data_excluding_label()),
                    data_info.get_data_label(),
                    n_clusters=4,
                )
                df_clusters = tsne_cluster.get_hard_clusters()
                st.dataframe(df_clusters)
                st.write(f"Evaluation score is {tsne_cluster.get_silhouette_score()}")

        elif option == "Model":
            st.subheader("Predictive Modeling")
            if st.checkbox("Logistic Regression"):
                lr_classification = LRModel(
                    data_info.get_data_excluding_label(), data_info.get_data_label()
                )
                st.write("Classification Report for Logistic Regression")
                st.dataframe(lr_classification.get_report())
                if st.button("Save this Model"):
                    lr_classification.save_the_model(file_paths["model_path"])
                    st.success("Model Saved Successfully")

            if st.checkbox("Logistic Regression with Principle Components"):
                pca_lr_classification = LRModel(
                    get_principle_components(data_info.get_data_excluding_label()),
                    data_info.get_data_label(),
                )
                st.dataframe(pca_lr_classification.get_report())
                if st.button("Save this Model"):
                    pca_lr_classification.save_the_model(file_paths["model_path"])
                    st.success("Model Saved Successfully")

            if st.checkbox("Logistic Regression with TSNE Components"):
                tsne_lr_classification = LRModel(
                    get_tsne_components(data_info.get_data_excluding_label()),
                    data_info.get_data_label(),
                )
                st.dataframe(tsne_lr_classification.get_report())
                if st.button("Save this Model"):
                    tsne_lr_classification.save_the_model(file_paths["model_path"])
                    st.success("Model Saved Successfully")


if __name__ == "__main__":
    display_user_interface()
