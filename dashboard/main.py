import yaml
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from visualiation import Visualization
from modeling import Clustering, ClassificationModels
with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)
matplotlib.use('Agg')
st.title('Crop Production Optimization System')


def check_data(df):
    st.dataframe(df.head(12))

    if st.checkbox("Display Shape"):
        st.write(df.shape)

    if st.checkbox("Check Null"):
        null_value = df.isnull().sum().sum()
        st.write("{null_value} null values found.".format(null_value=null_value))

    if st.checkbox("Check Column"):
        selected_column = st.multiselect('Select Desired Column', df.columns)
        if selected_column:
            st.dataframe(df[selected_column])

    if st.checkbox("Check object data types"):
        object_datas = df.select_dtypes('object')
        if not object_datas.empty:
            st.write(object_datas.head(3))
        else:
            st.write("No Object Data Found !")

    if st.checkbox("Check for data imbalance"):
        st.write(df['label'].value_counts())
        sns.countplot(x='label', data=df)
        st.pyplot(plt)


def main():
    data = st.file_uploader(label="", type=['csv', 'xlsx'])
    if data is not None:
        df = pd.read_csv(data)
        option = st.sidebar.selectbox('Select Option', file_paths['drop_down_list'])

        if option == 'Data Analysis':
            st.subheader("Understanding the Data")
            check_data(df)

        elif option == 'EDA':
            st.subheader("Exploratory Data Analysis")

            if st.checkbox("Data statistics"):
                dropdown_list = df.label.unique()
                dropdown_list = np.insert(dropdown_list, 0, 'Overall')
                selected_target = st.selectbox('Select Target', dropdown_list)
                if selected_target == "Overall":
                    st.write(df.describe())
                else:
                    st.write(df[df.label == selected_target].describe())

            st.subheader("Visualizations")
            visuals = Visualization()
            if st.checkbox("Distribution Plot"):
                selected_column = st.multiselect('Select the columns you want to visualize',
                                                 df.columns)
                if selected_column:
                    df = df[selected_column]
                    st.dataframe(df)
                fig = visuals.distribution_plot(df.drop(['label'], axis=1))
                st.pyplot(fig)

            if st.checkbox("Visualize for Optimal Number of Cluster"):
                fig = visuals.elbow_plot(df.drop(['label'], axis=1))
                st.pyplot(fig)

        elif option == 'Cluster Analysis':
            cluster = Clustering()
            st.subheader("Clustering")
            if st.checkbox("K-means clustering"):
                st.write("Hard Clustering List:")
                df_clusters = cluster.showclusters(df, n_clusters=4)
                st.dataframe(df_clusters)

        elif option == "Model":
            classification = ClassificationModels(df)
            st.subheader("Predictive Modeling")
            if st.checkbox("Logistic Regression"):
                model = classification.logistic_regression()
                fig, report = classification.evaluate_model_performance(model)
                st.pyplot(fig)
                st.write("Classification Report for Logistic Regression")
                st.dataframe(report)
                if st.button("Save this Model"):
                    classification.save_the_model(model, file_paths['model_path'])
                    st.success("Model Saved Successfully")

        # elif option == 'Feature Selection':
        #     cluster = Clustering()
        #     st.subheader("Clustering")
        #     if st.checkbox("K-means clustering"):
        #         st.write("Hard Clustering List:")
        #         df_clusters = cluster.showclusters(df, n_clusters=4)
        #         st.dataframe(df_clusters)


if __name__ == "__main__":
    main()
