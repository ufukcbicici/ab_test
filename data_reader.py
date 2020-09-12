import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from category_encoders import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import FunctionTransformer


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

    def __init__(self, filename, target_scaling, nan_elemination_ratio=0.5, test_ratio=0.1):
        self.mainDataFrame = pd.read_csv(filename)
        self.categoricalColumns = list([k for k, v in Dataset.column_descriptions.items()
                                        if v[0] == "Categorical" and (v[1] == "Independent Variable" or
                                                                      v[1] == "Group Type")])
        self.numericalColumns = list([k for k, v in Dataset.column_descriptions.items()
                                      if v[0] == "Numerical" and (v[1] == "Independent Variable" or
                                                                  "Pre-Test" in v[1])])
        self.numericalColumns.append("n1")
        self.targetColumns = list([k for k, v in Dataset.column_descriptions.items()
                                   if v[0] == "Numerical" and "Post-Test" in v[1]])
        self.targetScaling = target_scaling
        self.targetScalers = {}
        self.categoricalEncoders = {}
        self.encodedCategoricalVariables = {}
        self.targetVariables = {}
        self.independentVariables = {}
        self.variableImputers = {}
        self.trainIndices, self.testIndices = train_test_split(np.arange(self.mainDataFrame.shape[0]),
                                                               test_size=test_ratio)
        self.nanEliminationRatio = nan_elemination_ratio

    def scale_target_variables(self):
        for target_column in self.targetColumns:
            assert target_column in self.targetScaling
            scaler_type = self.targetScaling[target_column]
            if scaler_type == "identity":
                scaler = FunctionTransformer()
            elif scaler_type == "log":
                scaler = FunctionTransformer(func=np.log, inverse_func=np.exp)
            # Power Transform
            else:
                scaler = PowerTransformer()
            # Fit on the training samples
            _y = self.mainDataFrame[target_column].iloc[self.trainIndices]
            scaler.fit(_y[:, np.newaxis])
            # Be sure that the inverse transform works as expected
            _y_transformed = pd.Series(scaler.transform(self.mainDataFrame[target_column][:, np.newaxis])[:, 0])
            _y_back = scaler.inverse_transform(_y_transformed[:, np.newaxis])[:, 0]
            assert np.allclose(self.mainDataFrame[target_column], _y_back)
            # Scale all target variables; save them and the scaler
            self.targetVariables[target_column] = _y_transformed
            self.targetScalers[target_column] = scaler

    # If a column has more than #nan/elem_count ratio than self.nanEleminationRatio, exclude it from further processing
    def eliminate_columns(self):
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

    # We apply encoding to the target variables with "Target Encoding" approach. It both handles nan values and missing
    # values and avoids excessive number of dummy variables created with the 1-to-N one hot encoding method.
    def preprocess_categorical_variables(self):
        categorical_data = self.mainDataFrame[self.categoricalColumns]
        for target_column in self.targetColumns:
            target_encoder = TargetEncoder(cols=self.categoricalColumns)
            _X = categorical_data.iloc[self.trainIndices]
            _y = self.targetVariables[target_column].iloc[self.trainIndices]
            self.categoricalEncoders[target_column] = target_encoder
            # Fit on the training samples
            target_encoder.fit(X=_X, y=_y)
            # Transform the whole data
            self.encodedCategoricalVariables[target_column] = target_encoder.transform(X=categorical_data)

    # Before calling this method, we have converted all categorical variables. So technically, we can use the knn-imputer
    # using all independent variables right now.
    def preprocess_numerical_variables(self):
        numerical_variables = self.mainDataFrame[self.numericalColumns]
        # Use knn-Imputer for any missing data
        for target_column in self.targetColumns:
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            categorical_variables = self.encodedCategoricalVariables[target_column]
            all_independent_variables = pd.concat([categorical_variables, numerical_variables], axis=1)
            assert all_independent_variables.shape[0] == self.mainDataFrame.shape[0]
            # Fit on the training samples
            imputer.fit(all_independent_variables.iloc[self.trainIndices])
            # Transform the whole data
            all_independent_variables = imputer.transform(all_independent_variables)
            self.independentVariables[target_column] = all_independent_variables

    def data_exploration(self):
        # Categorical variables
        limit_ratio = 0.003
        # for cat_col in self.categoricalColumns:
        #     plt.figure(figsize=(10.0, 4.8))
        #     series = self.mainDataFrame[cat_col].value_counts(dropna=False)
        #     # Trim to fit into a single image, if there are too many categories
        #     if len(series) > 10:
        #         total_freq = series.sum()
        #         trimmed_series = series[series >= int(total_freq * limit_ratio)]
        #         others_total_freq = series[series < int(total_freq * limit_ratio)].sum()
        #         columns = [str(x) for x in trimmed_series.index.to_list()]
        #         values = trimmed_series.values.tolist()
        #         if len(series[series < int(total_freq * limit_ratio)]) > 0:
        #             columns.append("{0} others".format(len(series[series < int(total_freq * limit_ratio)])))
        #             values.append(others_total_freq)
        #     else:
        #         columns = [str(x) for x in series.index.to_list()]
        #         values = series.values.tolist()
        #     y_pos = np.arange(len(columns))
        #     plt.bar(y_pos, values, align='center', alpha=0.5)
        #     fontsize = "7" if "others" in columns[-1] else "12"
        #     plt.xticks(y_pos, columns, fontsize=fontsize, rotation=45)
        #     plt.ylabel('Frequency')
        #     plt.title("Categorical Column:{0}, {1} categories".format(cat_col, len(series)))
        #     plt.savefig("{0}_bars.png".format(cat_col))
        #     plt.show()

        num_columns = []
        num_columns.extend(self.numericalColumns)
        num_columns.extend(self.targetColumns)
        num_columns = set(num_columns)
        # descriptive_statistics = {
        #     "min": [],
        #     "max": [],
        #     "mean": [],
        #     "median": [],
        #     "std": [],
        #     "% of nan": []}
        # Descriptive statistics table
        # plt.figure(figsize=(10.0, 4.8))
        max_bin_count = 50
        very_large_threshold = 10e9
        very_small_threshold = -10e9

        def determine_histogram_bins(min_, max_):
            min_bin_length = (max_ - min_) / max_bin_count
            unit_size = 10 ** int(np.log10(min_bin_length))
            bin_length = int(((min_bin_length // unit_size) + 1) * unit_size)
            max_bin_limit = int(((max_val // bin_length) + 1) * bin_length)
            min_bin_limit = int((min_val // bin_length) * bin_length)
            bins_ = list(range(min_bin_limit, max_bin_limit + bin_length, bin_length))
            return bins_

        # unit_size = 200
        for num_col in num_columns:
            # fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.8))
            # fig, axes = plt.subplots(2, 1, figsize=(10.0, 4.8))
            plt.figure(figsize=(10.0, 4.8))
            series = self.mainDataFrame[num_col]
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
                bins = determine_histogram_bins(min_val, max_val)
                title = "Numerical column {0} histogram".format(num_col)
            else:
                trimmed_series = series.copy(deep=True)
                trimmed_series = trimmed_series[trimmed_series <= very_large_threshold]
                trimmed_series = trimmed_series[trimmed_series >= very_small_threshold]
                # trimmed_series[series < very_small_threshold] = 1.5 * very_small_threshold
                max_val = trimmed_series.max()
                min_val = trimmed_series.min()
                bins = determine_histogram_bins(min_val, max_val)
                title = "Numerical column {0} histogram. Showing %{1:.2f} of data.".format(
                    num_col,
                    100.0 * len(trimmed_series) / len(series))

            # ax = \
            ax = series.hist(bins=bins, grid=False, zorder=2, rwidth=0.9)
            ax.set_title(title)
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

            # df = pd.DataFrame(descriptive_statistics, index=None)
            # table(ax, df, loc='upper right', colWidths=[0.15] * len(descriptive_statistics),
            #       cellLoc="center", fontsize=10)
            # Descriptive statistics table
            # df = pd.DataFrame(descriptive_statistics)
            # df.plot(table=True, ax=ax)
            plt.show()
            print("X")

            # descriptive_statistics["min"].append(series.min())
            # descriptive_statistics["max"].append(series.max())
            # descriptive_statistics["mean"].append(series.mean())
            # descriptive_statistics["median"].append(series.median())
            # descriptive_statistics["std"].append(series.std())
            # descriptive_statistics["% of nan"].append(100.0 * series.isna().sum() / len(series))
        # ds = pd.DataFrame(descriptive_statistics, index=num_columns)
        # # ds.plot()
        # fig = plt.figure(figsize=(10.0, 4.8))
        # ax = fig.add_subplot(111)
        # y = [1, 2, 3, 4, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        # col_labels = ['col1', 'col2', 'col3']
        # row_labels = ['row1', 'row2', 'row3']
        # table_vals = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]

        # the_table = plt.table(cellText=ds.values,
        #                       colWidths=[0.25] * len(ds.columns),
        #                       rowLabels=ds.index,
        #                       colLabels=ds.columns,
        #                       cellLoc='center',
        #                       rowLoc='center',
        #                       loc='top')
        # the_table = plt.table(cellText=table_vals,
        #                       colWidths=[0.1] * 3,
        #                       rowLabels=row_labels,
        #                       colLabels=col_labels,
        #                       loc='center')
        # the_table.auto_set_font_size(False)
        # the_table.set_fontsize(24)
        # the_table.scale(4, 4)
        # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        # plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        # for pos in ['right', 'top', 'bottom', 'left']:
        #     plt.gca().spines[pos].set_visible(False)
        # plt.show()
        print("X")

    def prepare_dataset(self):
        self.eliminate_columns()
        self.scale_target_variables()
        self.preprocess_categorical_variables()
        self.preprocess_numerical_variables()

    # def data_exploration(self):
    #     knn_imputer = KNNImputer(n_neighbors=5)
    #     target_encoder = enc = TargetEncoder(cols=self.categoricalColumns)
    #     categorical_data = self.mainDataFrame[self.categoricalColumns]
    #     # fitted_data = knn_imputer.fit_transform(categorical_data)
    #     encoded_data = target_encoder.fit_transform(X=categorical_data, y=self.mainDataFrame["n13"])
    #     # encoded_data_2 = target_encoder.transform(X=[""])
    #     print("X")

    # for col in self.mainDataFrame.columns:
    #     assert col in Dataset.column_descriptions
    #     col_properties = Dataset.column_descriptions[col]
    #     col_type = col_properties[0]
    #     col_description = col_properties[1]
    #     if col_description == "Index":
    #         continue
    #     if col_type == "Categorical":
    #         categorical_value_distribution = self.mainDataFrame[col].value_counts(dropna=False)
    #         categorical_value_distribution.plot(kind='bar')
    #         plt.show()
    #         print("X")
