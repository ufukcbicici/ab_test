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

    def get_target_variable_scaler(self, y, target_column, group_type):
        pass

    def train_regressors(self):
        pass

