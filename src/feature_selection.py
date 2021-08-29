import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


class FeatureSelection:
    def __init__(self, data):
        self.data = data

    def correlation_analysis(self):
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.style.use('fivethirtyeight')
        corr = self.data.corr()
        sns.heatmap(corr, annot=True, cmap='viridis', linewidth=0.2)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        return plt

    def vif_analysis(self):
        vif_data = pd.DataFrame()
        vif_data["feature"] = self.data.columns
        vif_data["VIF"] = [variance_inflation_factor(self.data.values, i)
                           for i in range(len(self.data.columns))]
        return vif_data.style.background_gradient(cmap='viridis')

