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
from scipy.stats import t

# from pandas.plotting import table
from target_variable_scaler import HistogramScalerKBinsSupportingInf


class Recommender:
    def __init__(self, dataset, target_scaling):
        self.dataset = dataset
        self.targetScaling = target_scaling
        self.targetScalers = {}
        self.models = {}
        self.scoreScalers = {}
        # Prepare the dataset
        dataset.categoricalColumns.append("player_group")
        self.dataset.eliminate_nan_columns()

    def get_Xy(self, data_frame, target_column, group_type):
        data_frame = data_frame[data_frame.player_group == group_type]
        X = data_frame[self.dataset.independentVariables]
        y = data_frame[target_column]
        return X, y

    def get_target_variable_scaler(self, y, target_column, group_type=None):
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
        self.targetScalers[target_column] = scaler
        return scaler

    def train_regressors(self):
        main_data_frame = self.dataset.mainDataFrame.copy(deep=True)
        train_data = main_data_frame.iloc[self.dataset.trainIndices]
        test_data = main_data_frame.iloc[self.dataset.testIndices]
        # Train two models, for each target variable. Use all samples from both player groups
        for target_column in self.dataset.targetColumns:
            if target_column == "n13":
                continue
            # Training set
            X_train = train_data[self.dataset.independentVariables]
            y_train = train_data[target_column]
            # Test set
            X_test = test_data[self.dataset.independentVariables]
            y_test = test_data[target_column]
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
                "pca__n_components": [None],  # [3, 5, 7, 10, 15, 20, None],
                "rdf__n_estimators": [250],  # [100, 250, 500, 1000, 5000, 10000],
                "rdf__bootstrap": [True],
                "rdf__max_depth": [10, 20, 30]
            }]
            # Pipeline and Grid Search with K-Fold Cross Validation
            search = GridSearchCV(pipeline, param_grid, n_jobs=8, cv=10, verbose=10,
                                  scoring=["neg_mean_squared_error", "r2"], refit="neg_mean_squared_error")
            search.fit(X_train, target_scaler.transform(y_train[:, np.newaxis])[:, 0])
            print("**********For target:{0}**********".format(target_column))
            print("Best parameter (CV score=%0.3f):" % search.best_score_)
            print(search.best_params_)
            # Score the training and test sets
            best_model = search.best_estimator_
            for indices, data_type in zip([self.dataset.trainIndices, self.dataset.testIndices], ["Train", "Test"]):
                r2_scaled, r2_unscaled, mse_scaled, mse_unscaled = self.score_data_subset(target_column=target_column,
                                                                                          indices=indices)
                print("{0} Scaled R2:{1} Scaled MSE:{2} Unscaled R2:{3} Unscaled MSE:{4}".format(
                    data_type, r2_scaled, mse_scaled, r2_unscaled, mse_unscaled))
            # Fit a Z-Scaler to the predictions on the training set.
            y_predict = target_scaler.inverse_transform(best_model.predict(X=X_train)[:, np.newaxis])
            score_scaler = StandardScaler()
            score_scaler.fit(y_predict)
            self.models[target_column] = best_model
            self.scoreScalers[target_column] = score_scaler
            self.targetScalers[target_column] = target_scaler
            file_path = os.path.join("models", "unified_best_model_target_{0}_scale_{1}.sav".format(target_column,
                                                                                                    self.targetScaling[
                                                                                                        target_column]))
            self.save_model(file_path=file_path, model=best_model, target_scaler=target_scaler,
                            score_scaler=score_scaler)
            print("X")

    def save_model(self, file_path, model, target_scaler, score_scaler):
        model_file = open(file_path, "wb")
        model_dict = {"model": model, "target_scaler": target_scaler, "score_scaler": score_scaler}
        pickle.dump(model_dict, model_file)
        model_file.close()

    def load_models(self):
        for target_column in self.dataset.targetColumns:
            file_path = os.path.join("models",
                                     "unified_best_model_target_{0}_scale_{1}.sav".format(target_column,
                                                                                          self.targetScaling[
                                                                                              target_column]))
            assert os.path.isfile(file_path)
            model_file = open(file_path, "rb")
            model_dict = pickle.load(model_file)
            self.models[target_column] = model_dict["model"]
            self.targetScalers[target_column] = model_dict["target_scaler"]
            self.scoreScalers[target_column] = model_dict["score_scaler"]
            model_file.close()

    def score_data_subset(self, target_column, indices):
        data_subset = self.dataset.mainDataFrame.iloc[indices]
        X_ = data_subset[self.dataset.independentVariables]
        y_truth_unscaled = data_subset[target_column]
        y_truth_scaled = self.targetScalers[target_column].transform(y_truth_unscaled[:, np.newaxis])[:, 0]
        y_pred_scaled = self.models[target_column].predict(X=X_)
        y_pred_unscaled = self.targetScalers[target_column].inverse_transform(y_pred_scaled[:, np.newaxis])[:, 0]
        r2_scaled = r2_score(y_true=y_truth_scaled, y_pred=y_pred_scaled)
        r2_unscaled = r2_score(y_true=y_truth_unscaled, y_pred=y_pred_unscaled)
        mse_scaled = mean_squared_error(y_true=y_truth_scaled, y_pred=y_pred_scaled, squared=False)
        mse_unscaled = mean_squared_error(y_true=y_truth_unscaled, y_pred=y_pred_unscaled, squared=False)
        return r2_scaled, r2_unscaled, mse_scaled, mse_unscaled

    def score_csv_file(self, csv_file, lambda_):
        df = pd.read_csv(csv_file)
        # Assert that the file contains correct columns
        input_columns = set(df.columns)
        assert all([col in input_columns for col in self.dataset.independentVariables])
        # Select all relevant columns; create two copies for two groups
        df_list = []
        scores_list = []
        for group in ["A", "B"]:
            X = df[self.dataset.independentVariables].copy(deep=True)
            X["player_group"] = group
            # Predict regression targets, scaled
            y_engagement_pred_scaled = self.models["n13"].predict(X=X)
            y_money_pred_scaled = self.models["n14"].predict(X=X)
            # Unscale regression targets
            y_engagement_pred = self.targetScalers["n13"].inverse_transform(y_engagement_pred_scaled[:, np.newaxis])
            y_money_pred = self.targetScalers["n14"].inverse_transform(y_money_pred_scaled[:, np.newaxis])
            # Scale with a standard Z-Scaler such that both predicted values are in the same scale; so we can mix them
            # together in a weighted sum.
            score_engagement = self.scoreScalers["n13"].transform(y_engagement_pred)
            score_money = self.scoreScalers["n13"].transform(y_money_pred)
            score = lambda_ * score_engagement + (1.0 - lambda_) * score_money
            df_x = pd.DataFrame(data=np.concatenate(
                [y_engagement_pred, y_money_pred, score_engagement, score_money, score], axis=1),
                                columns=["y_engagement_pred_{0}".format(group),
                                         "y_money_pred_{0}".format(group),
                                         "score_engagement_{0}".format(group),
                                         "score_money_{0}".format(group),
                                         "score_{0}".format(group)])
            df_list.append(df_x)
            scores_list.append(score)
        score_matrix = np.concatenate(scores_list, axis=1)
        selection = np.argmax(score_matrix, axis=1).astype(np.bool)
        selected_groups = np.where(selection, "B", "A")
        final_df = pd.concat([df, df_list[0], df_list[1]], axis=1)
        final_df["recommended_group"] = selected_groups
        result_path = csv_file[:-4] + "_recommendations.csv"
        final_df.to_csv(result_path, index=False)
        print("X")

    # Calculate a global design recommendation
    def determine_global_recommendation(self):
        # Calculate the recommendations for every user in our original data
        self.score_csv_file(csv_file="toydata_mltest.csv", lambda_=0.5)
        # Open the recommendations file
        df = pd.read_csv("toydata_mltest_recommendations.csv")
        # For every player (row), get the predicted post-test engagement and monetization metrics for both groups.
        post_test_engagements_A = df["y_engagement_pred_A"]
        post_test_monetizations_A = df["y_money_pred_A"]
        post_test_engagements_B = df["y_engagement_pred_B"]
        post_test_monetizations_B = df["y_money_pred_B"]
        # We are going to apply correlated t-tests to check whether there are statistically
        # significant differences between the calculated post-test engagement and monetization means.
        degree_of_freedoms = df.shape[0] - 1
        # Test for significance in the engagement means between groups A and B
        dif_engagement = post_test_engagements_A - post_test_engagements_B
        # Is the difference in the means significantly different from zero?
        M_engagement = dif_engagement.mean()
        mu_engagement = 0
        s_A = np.std(post_test_engagements_A, ddof=1)
        s_B = np.std(post_test_engagements_B, ddof=1)
        r_AB = np.corrcoef(post_test_engagements_A, post_test_engagements_B)[0, 1]
        s_M = np.sqrt(((s_A**2.0 + s_B**2.0) - (2*r_AB*s_A*s_B)) / df.shape[0])
        t_statistic = (np.abs(M_engagement) - mu_engagement) / s_M
        two_tailed_probability = (1.0 - t.cdf(t_statistic, degree_of_freedoms)) * 2.0
        # is_significant = two_tailed_probability >= 0.05
        print("Post test engagement is {0} larger for group {1}. P value is {2}".format(
            np.abs(M_engagement), "A" if M_engagement > 0 else "B", two_tailed_probability))

        # Test for significance in the monetization means between groups A and B
        dif_money = post_test_monetizations_A - post_test_monetizations_B
        # Is the difference in the means significantly different from zero?
        M_money = dif_money.mean()
        mu_money = 0
        s_A = np.std(post_test_monetizations_A, ddof=1)
        s_B = np.std(post_test_monetizations_B, ddof=1)
        r_AB = np.corrcoef(post_test_monetizations_A, post_test_monetizations_B)[0, 1]
        s_M = np.sqrt(((s_A**2.0 + s_B**2.0) - (2*r_AB*s_A*s_B)) / df.shape[0])
        t_statistic = (np.abs(M_money) - mu_money) / s_M
        two_tailed_probability = (1.0 - t.cdf(t_statistic, degree_of_freedoms)) * 2.0
        # is_significant = two_tailed_probability >= 0.05
        print("Post test monetization is {0} larger for group {1}. P value is {2}".format(
              np.abs(M_money), "A" if M_money > 0 else "B", two_tailed_probability))

