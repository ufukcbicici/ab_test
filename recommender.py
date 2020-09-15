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
from sklearn.metrics import mean_squared_error, r2_score

# from pandas.plotting import table
from target_variable_scaler import HistogramScalerKBinsSupportingInf


class Recommender:
    def __init__(self, dataset, target_scaling):
        self.dataset = dataset
        self.targetScaling = target_scaling
        self.targetScalers = {}

    def get_Xy(self, data_frame, target_column, group_type):
        data_frame = data_frame[data_frame.player_group == group_type]
        X = data_frame[self.dataset.independentVariables]
        y = data_frame[target_column]
        return X, y

    def get_target_variable_scaler(self, target_column, group_type, y):
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
        # _y_transformed = pd.Series(scaler.transform(self.dataset.mainDataFrame[target_column][:, np.newaxis])[:, 0])
        # _y_back = scaler.inverse_transform(_y_transformed[:, np.newaxis])[:, 0]
        # assert np.allclose(self.dataset.mainDataFrame[target_column], _y_back)
        self.targetScalers[(target_column, group_type)] = scaler
        return scaler

    def train_regressors(self):
        self.dataset.eliminate_nan_columns()
        main_data_frame = self.dataset.mainDataFrame.copy(deep=True)
        group_types = main_data_frame["player_group"].unique()
        train_data = main_data_frame.iloc[self.dataset.trainIndices]
        test_data = main_data_frame.iloc[self.dataset.testIndices]
        for target_column in self.dataset.targetColumns:
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
                target_scaler = self.get_target_variable_scaler(target_column=target_column, group_type=group_type,
                                                                y=y_train)
                # 1)
                # We apply encoding to the target variables with "Target Encoding" approach.
                # It both handles nan values and missing
                # values and avoids excessive number of dummy variables created with the 1-to-N one hot encoding method.
                # 2)
                # The numerical feature "n10" contains "inf". We cannot process or normalize this value.
                # We instead are going to quantize it.
                transformers_list = [
                    ("categorical_target_encoder",
                     TargetEncoder(cols=self.dataset.categoricalColumns), self.dataset.categoricalColumns),
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
                    "rdf__max_depth": [10, 20, 30]
                }]
                # Pipeline and Grid Search with K-Fold Cross Validation
                search = GridSearchCV(pipeline, param_grid, n_jobs=8, cv=10, verbose=10,
                                      scoring=["neg_mean_squared_error", "r2"], refit="neg_mean_squared_error")
                search.fit(X_train, target_scaler.transform(y_train[:, np.newaxis])[:, 0])
                print("**********For target:{0} and group_type:{1}**********".format(target_column, group_type))
                print("Best parameter (CV score=%0.3f):" % search.best_score_)
                print(search.best_params_)
                # Score the training and test sets
                best_model = search.best_estimator_
                for X_, y_unscaled, data_type in zip([X_train, X_test], [y_train, y_test], ["Train", "Test"]):
                    y_scaled = target_scaler.transform(y_unscaled[:, np.newaxis])[:, 0]
                    y_pred_scaled = best_model.predict(X=X_)
                    y_pred_unscaled = target_scaler.inverse_transform(y_pred_scaled[:, np.newaxis])[:, 0]
                    r2_scaled = r2_score(y_true=y_scaled, y_pred=y_pred_scaled)
                    mse_scaled = mean_squared_error(y_true=y_scaled, y_pred=y_pred_scaled, squared=False)
                    r2_unscaled = r2_score(y_true=y_unscaled, y_pred=y_pred_unscaled)
                    mse_unscaled = mean_squared_error(y_true=y_unscaled, y_pred=y_pred_unscaled, squared=False)
                    print("{0} Scaled R2:{1} Scaled MSE:{2} Unscaled R2:{3} Unscaled MSE:{4}".format(
                        data_type, r2_scaled, mse_scaled, r2_unscaled, mse_unscaled))
                # Save all model related objects
                model_file = open(os.path.join("models", "best_model_target_{0}_group_type_{1}.sav"
                                               .format(target_column, group_type)), "wb")
                model_dict = {"model": best_model, "target_scaler": target_scaler}
                pickle.dump(model_dict, model_file)
                model_file.close()
                print("X")
