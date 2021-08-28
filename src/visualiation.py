import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.style.use('seaborn')


class Visualization:
    def __init__(self, data):
        self.data = data

    def pair_plot(self):
        sns.pairplot(self.data, diag_kind='kde')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        return plt

    def distribution_plot(self):
        fig, axes = plt.subplots(3, 3)
        axes = axes.ravel()
        for col, ax in zip(self.data.columns, axes):
            sns.histplot(data=self.data[col], kde=True, stat='count', ax=ax)
        fig.tight_layout()
        return plt

    def box_plot(self):
        fig, axes = plt.subplots(3, 3)
        axes = axes.ravel()
        for col, ax in zip(self.data.columns, axes):
            sns.boxplot(data=self.data[col], ax=ax)
        fig.tight_layout()
        return plt

