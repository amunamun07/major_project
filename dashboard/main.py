import yaml
import pandas as pd
import streamlit as st

from clustering import Clustering
from data_info import DataInformation
from visualization import Visualization
from modeling import LRModel

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)

st.set_option("deprecation.showPyplotGlobalUse", False)
st.title("Crop Production Dashboard")


def main():
    data = st.file_uploader(label="", type=["csv", "xlsx"])
    if data is not None:
        df = pd.read_csv(data)
        option = st.sidebar.selectbox("Select Option", file_paths["drop_down_list"])

        data_info = DataInformation(df)
        visuals = Visualization()
        cluster = Clustering(data_info.get_data_excluding_label(), data_info.get_data_label())
        lr_classification = LRModel(data_info.get_data_excluding_label(), data_info.get_data_label())

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
                    st.dataframe(
                        data_info.get_selected_column(selected_column)
                    )

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
            if st.checkbox("Distribution Plot"):
                selected_column = st.multiselect(
                    "Select the columns you want to visualize", data_info.get_dataset_columns()
                )
                if selected_column:
                    returned_data = data_info.get_selected_column(selected_column)
                    st.dataframe(returned_data)
                    visuals.plot_data_distribution(data_info.get_data_excluding_label(only_include=returned_data))
                else:
                    visuals.plot_data_distribution(data_info.get_data_excluding_label())
                st.pyplot()

            if st.checkbox("Find Optimal Number of Cluster"):
                visuals.plot_elbow_graph(data_info.get_data_excluding_label())
                st.pyplot()

        elif option == "Cluster Analysis":
            st.subheader("Clustering")
            if st.checkbox("K-means clustering"):
                df_clusters = cluster.show_clusters(n_clusters=4)
                st.dataframe(df_clusters)

        elif option == "Model":
            st.subheader("Predictive Modeling")
            if st.checkbox("Logistic Regression"):
                fig, report = lr_classification.evaluate_model_performance()
                st.pyplot(fig)
                st.write("Classification Report for Logistic Regression")
                st.dataframe(report)
                if st.button("Save this Model"):
                    lr_classification.save_the_model(file_paths["model_path"])
                    st.success("Model Saved Successfully")


if __name__ == "__main__":
    main()
