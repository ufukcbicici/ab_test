import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import KNNImputer
from category_encoders import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import os
import pickle
from sklearn.metrics import mean_squared_error

# from pandas.plotting import table
from target_variable_scaler import HistogramScalerKBinsSupportingInf


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
        self.targetScaling = target_scaling
        self.targetScalers = {}
        self.categoricalEncoders = {}
        self.encodedCategoricalVariables = {}
        self.targetVariables = {}
        self.independentVariables = []
        self.variableImputers = {}
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

    def get_target_variable_scaler(self, target_column, y):
        assert target_column in self.targetScaling
        scaler_type = self.targetScaling[target_column]
        if scaler_type == "identity":
            scaler = FunctionTransformer()
        elif scaler_type == "log":
            scaler = FunctionTransformer(func=np.log, inverse_func=np.exp)
        # Power Transform
        else:
            scaler = PowerTransformer()
        # Fit the scaler
        scaler.fit(y[:, np.newaxis])
        # Be sure that the inverse transform works as expected
        _y_transformed = pd.Series(scaler.transform(self.mainDataFrame[target_column][:, np.newaxis])[:, 0])
        _y_back = scaler.inverse_transform(_y_transformed[:, np.newaxis])[:, 0]
        assert np.allclose(self.mainDataFrame[target_column], _y_back)
        self.targetScalers[target_column] = scaler
        return scaler

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

    def data_exploration(self):
        # Categorical variables
        limit_ratio = 0.003
        for cat_col in self.categoricalColumns:
            plt.figure(figsize=(10.0, 4.8))
            series = self.mainDataFrame[cat_col].value_counts(dropna=False)
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
            plt.title("Categorical Column:{0}, {1} categories".format(cat_col, len(series)))
            plt.savefig("{0}_bars.png".format(cat_col))
            plt.show()

        num_columns = []
        num_columns.extend(self.numericalColumns)
        num_columns.extend(self.targetColumns)
        num_columns = set(num_columns)
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

        for num_col in num_columns:
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
            plt.savefig("{0}_hist.png".format(num_col))
            plt.show()
        print("X")

    def get_Xy(self, data_frame, target_column, group_type):
        data_frame = data_frame[data_frame.player_group == group_type]
        X = data_frame[self.independentVariables]
        y = data_frame[target_column]
        return X, y

    def prepare_dataset_v2(self):
        self.eliminate_nan_columns()
        group_types = self.mainDataFrame["player_group"].unique()
        train_data = self.mainDataFrame.iloc[self.trainIndices]
        test_data = self.mainDataFrame.iloc[self.testIndices]
        for target_column in self.targetColumns:
            # y = self.targetVariables[target_column].iloc[self.trainIndices]
            for group_type in group_types:
                # Training set
                X_train, y_train = self.get_Xy(data_frame=train_data, target_column=target_column,
                                               group_type=group_type)
                # Test set
                X_test, y_test = self.get_Xy(data_frame=test_data, target_column=target_column,
                                             group_type=group_type)
                # Scale the target variables. We see in data exploration that the target variables have large
                # standard deviations. To fit the regressor easier, we need to apply an invertible transform to the
                # corresponding target variables: n13 (post-test engagement time) and n14 (post-test monetization).
                target_scaler = self.get_target_variable_scaler(target_column=target_column, y=y_train)
                # 1)
                # We apply encoding to the target variables with "Target Encoding" approach.
                # It both handles nan values and missing
                # values and avoids excessive number of dummy variables created with the 1-to-N one hot encoding method.
                # 2)
                # The numerical feature "n10" contains "inf". We cannot process or normalize this value.
                # We instead are going to quantize it.
                transformers_list = [
                    ("categorical_target_encoder",
                     TargetEncoder(cols=self.categoricalColumns), self.categoricalColumns),
                    ("n10_discretizer",
                     HistogramScalerKBinsSupportingInf(columns=["n10"], bin_count=5), ["n10"]),
                ]
                transformer = ColumnTransformer(transformers=transformers_list, remainder="passthrough")
                # For any potential missing value; we apply k-nn imputer (after the column transformer is applied,
                # all of our data will be numerical).
                # We are going to then normalize the features and then apply PCA for dimensionality reduction.
                # We are going to use RandomForestRegressor since it is fast to train compared
                # to Gradient Boosting Machines with a small compromise on the accuracy.
                pipeline = Pipeline(steps=[
                    ("column_transformer", transformer),
                    ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
                    ("scaler", StandardScaler()),
                    ("pca", PCA()),
                    ("rdf", RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0))
                ])
                # Param grid:
                param_grid = [{
                    "pca__n_components": [5, 10, None],  # [3, 5, 7, 10, 15, 20, None],
                    "rdf__n_estimators": [50, 100, 250],  # [100, 250, 500, 1000, 5000, 10000],
                    "rdf__bootstrap": [False, True],
                    "rdf__max_depth": [5, 10, 15, 20, 25, 30]
                }]
                model = TransformedTargetRegressor(regressor=pipeline, transformer=target_scaler)
                # Pipeline and Grid Search with K-Fold Cross Validation
                search = GridSearchCV(model, param_grid, n_jobs=4, cv=10, verbose=10,
                                      scoring=["neg_mean_squared_error", "r2"], refit="neg_mean_squared_error")
                search.fit(X_train, y_train)
                print("**********For target:{0} and group_type:{1}**********".format(target_column, group_type))
                print("Best parameter (CV score=%0.3f):" % search.best_score_)
                print(search.best_params_)
                # Score the training and test sets
                best_model = search.best_estimator_
                for X_, y_, data_type in zip([X_train, X_test], [y_train, y_test], ["Train", "Test"]):
                    r2_score = best_model.score(X=X_, y=y_)
                    y_pred = best_model.predict(X=X_)
                    mse_ = mean_squared_error(y_, y_pred, squared=False)
                    print("{0} R2:{1} Train MSE:{2}".format(data_type, r2_score, mse_))
                # Save all model related objects
                model_file = open(os.path.join("models", "best_model_target_{0}_group_type_{1}.sav"
                                               .format(target_column, group_type)), "wb")
                model_dict = {"model": best_model}
                pickle.dump(model_dict, model_file)
                model_file.close()
                print("X")
