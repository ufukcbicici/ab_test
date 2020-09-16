import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# from pandas.plotting import table


class Dataset:
    column_descriptions = {
        "id": ("Categorical", "Index"),
        "c1": ("Categorical", "Independent Variable"),
        "c2": ("Categorical", "Independent Variable"),
        "c3": ("Categorical", "Independent Variable"),
        "c4": ("Categorical", "Independent Variable"),
        "c5": ("Categorical", "Independent Variable"),
        "c6": ("Categorical", "Independent Variable"),
        "player_group": ("Categorical", "Group Type"),
        "n1": ("Numerical", "Pre-Test 30 days Engagement Intensity"),
        "n13": ("Numerical", "Post-Test 7 days Engagement Intensity"),
        "n14": ("Numerical", "Post-Test 7 days Monetization Metric"),
        "n2": ("Numerical", "Independent Variable"),
        "n3": ("Numerical", "Independent Variable"),
        "n4": ("Numerical", "Independent Variable"),
        "n5": ("Numerical", "Independent Variable"),
        "n6": ("Numerical", "Independent Variable"),
        "n7": ("Numerical", "Independent Variable"),
        "n8": ("Numerical", "Independent Variable"),
        "n9": ("Numerical", "Independent Variable"),
        "n10": ("Numerical", "Independent Variable"),
        "n11": ("Numerical", "Independent Variable"),
        "n12": ("Numerical", "Independent Variable")
    }

    def __init__(self, filename, nan_elemination_ratio=0.5, test_ratio=0.1):
        # Set numpy framework such that train/test split is repeatable
        np.random.seed(42)
        self.mainDataFrame = pd.read_csv(filename)
        self.categoricalColumns = list([k for k, v in Dataset.column_descriptions.items()
                                        if v[0] == "Categorical" and v[1] == "Independent Variable"])
        self.numericalColumns = list([k for k, v in Dataset.column_descriptions.items()
                                      if v[0] == "Numerical" and (v[1] == "Independent Variable" or
                                                                  "Pre-Test" in v[1])])
        self.targetColumns = list([k for k, v in Dataset.column_descriptions.items()
                                   if v[0] == "Numerical" and "Post-Test" in v[1]])
        self.independentVariables = []
        self.trainIndices = None
        self.testIndices = None
        self.nanEliminationRatio = nan_elemination_ratio
        # Load or create train / test indices
        file_path = os.path.join("models", "train_test_indices.sav")
        if os.path.isfile(file_path):
            f = open(file_path, "rb")
            indices_dict = pickle.load(f)
            self.trainIndices = indices_dict["train"]
            self.testIndices = indices_dict["test"]
            f.close()
        else:
            self.trainIndices, self.testIndices = train_test_split(np.arange(self.mainDataFrame.shape[0]),
                                                                   test_size=test_ratio)
            indices_dict = {"train": self.trainIndices, "test": self.testIndices}
            f = open(file_path, "wb")
            pickle.dump(indices_dict, f)
            f.close()

    # If a column has more than #nan/elem_count ratio than self.nanEleminationRatio, exclude it from further processing
    def eliminate_nan_columns(self):
        def get_eligible_columns(col_list):
            eligible_columns = []
            for col_name in col_list:
                nan_ratio = self.mainDataFrame[col_name].iloc[self.trainIndices].isna().sum() / \
                            self.mainDataFrame[col_name].iloc[self.trainIndices].shape[0]
                print("Column {0} nan ratio:{1}".format(col_name, nan_ratio))
                if nan_ratio < self.nanEliminationRatio:
                    print("Column {0} is selected".format(col_name))
                    eligible_columns.append(col_name)
                else:
                    print("Column {0} is discarded".format(col_name))
            return eligible_columns

        self.categoricalColumns = get_eligible_columns(self.categoricalColumns)
        self.numericalColumns = get_eligible_columns(self.numericalColumns)
        self.independentVariables = []
        self.independentVariables.extend(self.categoricalColumns)
        self.independentVariables.extend(self.numericalColumns)

    @staticmethod
    def draw_categorical_data_histogram(series, limit_ratio=0.003):
        # Trim to fit into a single image, if there are too many categories
        if len(series) > 10:
            total_freq = series.sum()
            trimmed_series = series[series >= int(total_freq * limit_ratio)]
            others_total_freq = series[series < int(total_freq * limit_ratio)].sum()
            columns = [str(x) for x in trimmed_series.index.to_list()]
            values = trimmed_series.values.tolist()
            if len(series[series < int(total_freq * limit_ratio)]) > 0:
                columns.append("{0} others".format(len(series[series < int(total_freq * limit_ratio)])))
                values.append(others_total_freq)
        else:
            columns = [str(x) for x in series.index.to_list()]
            values = series.values.tolist()
        y_pos = np.arange(len(columns))
        plt.bar(y_pos, values, align='center', alpha=0.5)
        fontsize = "7" if "others" in columns[-1] else "12"
        plt.xticks(y_pos, columns, fontsize=fontsize, rotation=45)
        plt.ylabel('Frequency')

    @staticmethod
    def determine_histogram_bins(bin_count, min_val, max_val):
        _range = max_val - min_val
        # min_limit = min_val - 0.0001*min_val
        # max_limit = max_val + 0.0001*max_val
        bin_length = (max_val - min_val) / bin_count
        bins_ = np.arange(min_val, max_val + bin_length, bin_length)
        # min_bin_length = (max_val - min_val) / max_bin_count
        # unit_size = 10 ** int(np.log10(min_bin_length))
        # bin_length = ((min_bin_length // unit_size) + 1) * unit_size
        # if bin_length > 1.0:
        #     bin_length = int(bin_length)
        #     max_bin_limit = int(((max_val // bin_length) + 1) * bin_length)
        #     min_bin_limit = int((min_val // bin_length) * bin_length)
        # else:
        #     max_bin_limit = ((max_val // bin_length) + 1) * bin_length
        #     min_bin_limit = (min_val // bin_length) * bin_length
        # bins_ = np.arange(min_bin_limit, max_bin_limit + bin_length, bin_length)
        return bins_

    @staticmethod
    def draw_numerical_data_histogram(ax, num_col, series,
                                      very_small_threshold, very_large_threshold,
                                      max_bin_count, post_fix=None):
        descriptive_statistics = {
            "min": [series.min()],
            "max": [series.max()],
            "mean": [series.mean()],
            "median": [series.median()],
            "std": [series.std()],
            "% of nan": [100.0 * series.isna().sum() / len(series)]}
        max_val = descriptive_statistics["max"][0]
        min_val = descriptive_statistics["min"][0]
        if very_small_threshold < min_val and max_val < very_large_threshold:
            bins = Dataset.determine_histogram_bins(max_bin_count, min_val, max_val)
            title = "Numerical column {0} histogram".format(num_col)
        else:
            trimmed_series = series.copy(deep=True)
            trimmed_series = trimmed_series[trimmed_series <= very_large_threshold]
            trimmed_series = trimmed_series[trimmed_series >= very_small_threshold]
            # trimmed_series[series < very_small_threshold] = 1.5 * very_small_threshold
            max_val = trimmed_series.max()
            min_val = trimmed_series.min()
            bins = Dataset.determine_histogram_bins(max_bin_count, min_val, max_val)
            title = "Numerical column {0} histogram. Showing %{1:.2f} of data.".format(
                num_col,
                100.0 * len(trimmed_series) / len(series))
        ax = series.hist(bins=bins, grid=False, zorder=2, rwidth=0.9, ax=ax)
        ax.set_title(title if post_fix is None else title + " " + post_fix)
        vals = ax.get_yticks()
        for tick in vals:
            ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
        # Set y-axis label
        ax.set_ylabel("Frequency", labelpad=20, weight='bold', size=12)
        plt.xticks(bins, fontsize=8, rotation=45)
        tpls = [(k, "{0:.2f}".format(v[0])) for k, v in descriptive_statistics.items()]
        plt.table(cellText=[[tpl[1] for tpl in tpls]],
                  colWidths=[0.15] * len(descriptive_statistics),
                  rowLabels=None,
                  colLabels=[tpl[0] for tpl in tpls],
                  cellLoc='center',
                  rowLoc='center',
                  loc='upper right')

    def plot_categorical_variables(self):
        for cat_col in self.categoricalColumns:
            plt.figure(figsize=(10.0, 6))
            # Draw histograms for the whole data, group A and group B.
            ax = plt.subplot(3, 1, 1)
            series = self.mainDataFrame[cat_col].value_counts(dropna=False)
            plt.title("Categorical Column:{0}, {1} categories".format(cat_col, len(series)))
            Dataset.draw_categorical_data_histogram(series=series)
            ax = plt.subplot(3, 1, 2)
            series = self.mainDataFrame[self.mainDataFrame.player_group == "A"][cat_col].value_counts(dropna=False)
            ax.set_title("Player Group A")
            Dataset.draw_categorical_data_histogram(series=series)
            ax = plt.subplot(3, 1, 3)
            ax.set_title("Player Group B")
            series = self.mainDataFrame[self.mainDataFrame.player_group == "B"][cat_col].value_counts(dropna=False)
            Dataset.draw_categorical_data_histogram(series=series)
            plt.tight_layout()
            plt.savefig("{0}_bars.png".format(cat_col))
            plt.show()
            print("X")

    def plot_numerical_variables(self):
        num_columns = []
        num_columns.extend(self.numericalColumns)
        num_columns.extend(self.targetColumns)
        num_columns = set(num_columns)
        max_bin_count = 50
        very_large_threshold = 10e9
        very_small_threshold = -10e9
        for num_col in num_columns:
            plt.figure(figsize=(10.0, 8))
            ax = plt.subplot(3, 1, 1)
            series = self.mainDataFrame[num_col]
            Dataset.draw_numerical_data_histogram(ax=ax, num_col=num_col, series=series,
                                                  very_small_threshold=very_small_threshold,
                                                  very_large_threshold=very_large_threshold,
                                                  max_bin_count=max_bin_count)
            ax = plt.subplot(3, 1, 2)
            series = self.mainDataFrame[self.mainDataFrame.player_group == "A"][num_col]
            Dataset.draw_numerical_data_histogram(ax=ax, num_col=num_col, series=series,
                                                  very_small_threshold=very_small_threshold,
                                                  very_large_threshold=very_large_threshold,
                                                  max_bin_count=max_bin_count,
                                                  post_fix="(Player Group A)")
            ax = plt.subplot(3, 1, 3)
            series = self.mainDataFrame[self.mainDataFrame.player_group == "B"][num_col]
            Dataset.draw_numerical_data_histogram(ax=ax, num_col=num_col, series=series,
                                                  very_small_threshold=very_small_threshold,
                                                  very_large_threshold=very_large_threshold,
                                                  max_bin_count=max_bin_count,
                                                  post_fix="(Player Group B)")
            plt.tight_layout()
            plt.savefig("{0}_hist.png".format(num_col))
            plt.show()
            print("X")

    def data_exploration(self):
        self.plot_categorical_variables()
        self.plot_numerical_variables()
