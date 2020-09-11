# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
#
#
# class TargetScaler(BaseEstimator, TransformerMixin):
#
#     def __init__(self, scaling="identity"):
#         self.scaling = scaling
#
#     def transform(self, X, y=None):
#         if self.scaling not in {"identity", "log", "power"}:
#             raise ValueError("Unexpected scaling type:{0}".format(self.scaling))
#
#         if self.scaling == "identity":
#             X_hat = X.copy()
#         elif self.scaling == "log":
#
#
#         return do_something_to(X, self.vars)  # where the actual feature extraction happens
#
#     def fit(self, X, y=None):
#         return self  # generally does nothing