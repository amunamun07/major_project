import yaml
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from cleaning import DataCleaning
from feature_selection import FeatureSelection
from visualiation import Visualization
from modeling import ModelBuilding
from feature_extraction import FeatureExtraction
with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)
matplotlib.use('Agg')
st.title('HR System')


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


def get_data_for_visualization(df_visualization):
    group_of_employee = ['All', 'Employees who left',
                         'Employees who have not left',
                         'Employees who are Valued',
                         'Employees who were valued but left']

    selected_group = st.selectbox('Select Group', group_of_employee)
    if selected_group == 'Employees who left':
        df_visualization = df_visualization[df_visualization['left'] == 1]
    elif selected_group == 'Employees who have not left':
        df_visualization = df_visualization[df_visualization['left'] == 0]
    elif selected_group == 'Employees who are Valued':
        df_visualization = df_visualization[(df_visualization['last_evaluation'] > 0.7)
                                            | (df_visualization['time_spend_company'] > 4)
                                            | (df_visualization['promotion_last_5years'] > 1)
                                            | (df_visualization['number_project'] > 4)
                                            ]
    elif selected_group == 'Employees who were valued but left':
        df_visualization = df_visualization[(df_visualization['last_evaluation'] > 0.8)
                                            | (df_visualization['time_spend_company'] > 4)
                                            | (df_visualization['promotion_last_5years'] > 1)
                                            | (df_visualization['number_project'] > 5)
                                            ]
        df_visualization = df_visualization[df_visualization['left'] == 1]

    return df_visualization


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


    #         df = data_cleaning.get_cleaned_data(df)
    #
    #         st.subheader("1. Descriptive Statistics")
    #         if st.checkbox('Show Data Statistics'):
    #             st.write(df.describe())
    #
    #         st.subheader("2. Visualizations")
    #         df_visualization = get_data_for_visualization(df)
    #
    #         selected_column = st.multiselect('Select the columns you want to visualize',
    #                                          df_visualization.columns)
    #         if selected_column:
    #             df_visualization = df_visualization[selected_column]
    #             st.dataframe(df_visualization)
    #
    #         visuals = Visualization(df_visualization)
    #         if st.checkbox('Show Data Distribution'):
    #             fig = visuals.distribution_plot()
    #             st.pyplot(fig)
    #
    #         if st.checkbox('Show Pair Distribution'):
    #             if not len(df_visualization.columns) == 2:
    #                 st.write("Pair plot is only possible for pairs of column."
    #                          "Please select two columns.")
    #             else:
    #                 plt = visuals.pair_plot()
    #                 st.pyplot(plt)
    #
    #         # if st.checkbox('Show Outlier Distribution'):
    #         #     fig = visuals.box_plot()
    #         #     st.pyplot(fig)
    #
    #     elif option == 'Feature Selection':
    #         st.subheader("Feature Selection")
    #         df = data_cleaning.get_cleaned_data(df)
    #
    #         st.subheader("1. Correlation Check")
    #         feature_selection = FeatureSelection(df)
    #         if st.checkbox('Check Heatmap'):
    #             st.pyplot(feature_selection.correlation_analysis())
    #
    #         st.subheader("2. Multi Collinearity Check")
    #         if st.checkbox('Check VIF'):
    #             data = feature_selection.vif_analysis()
    #             st.write(data)
    #
    #     elif option == 'Feature Extraction':
    #         st.subheader("Feature Selection")
    #         df = data_cleaning.get_cleaned_data(df)
    #         feature_extraction = FeatureExtraction(df)
    #         st.subheader("1. PCA")
    #         if st.checkbox('Apply PCA'):
    #             selected_target = st.selectbox('Select Target', ['Employee Who left',
    #                                                              'Employee Who did not leave'])
    #             if selected_target == 'Employee Who left':
    #                 targets = [1]
    #                 colors = ["b"]
    #             else:
    #                 targets = [0]
    #                 colors = ["r"]
    #             feature_extraction.visualize_pca(targets, colors)
    #         st.subheader("2. TSNE")
    #         if st.checkbox('Apply TSNE'):
    #             feature_extraction.tsne_analysis()
    #
    #     elif option == 'model':
    #         st.subheader("Model")
    #         st.subheader("1. K-means analysis on data")
    #         model = ModelBuilding(df)
    #         fig = model.plot_elbow()
    #         st.pyplot(fig)
    #         model.k_means_clustering(n_clusters=5)
    #
    # else:
    #     st.write("Please upload your HR dataset here! ")


if __name__ == "__main__":
    main()
