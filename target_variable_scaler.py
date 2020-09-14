import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HistogramScalerKBinsSupportingInf(BaseEstimator, TransformerMixin):

    def __init__(self, columns, bin_count):
        self.columns = columns
        self.bin_count = bin_count
        self.intervalEdges = {}

    # @staticmethod
    # def get_bin(x, col, data_frame, edges):
    #     s = 0
    #     for idx in range(len(edges)):
    #         if x < edges[idx]:
    #             frame_dict["{0}_discrete{1}".format(col, idx)].append(1)
    #         else:
    #             frame_dict["{0}_discrete{1}".format(col, idx)].append(0)

    def transform(self, X, y=None):
        # Support only pandas Dataframes
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Dataframe is expected.")
        new_column_names = []
        discretized_data = []
        for col in self.columns:
            X_col = np.zeros(shape=(X[col].shape[0], self.bin_count))
            bin_indices = np.digitize(X[col], self.intervalEdges[col], right=True) - 1
            X_col[np.arange(X_col.shape[0]), bin_indices] = 1
            discretized_data.append(X_col)
            new_column_names.extend(["{0}_discrete{1}".format(col, idx) for idx in range(self.bin_count)])
        X_arr = np.concatenate(discretized_data, axis=1)
        X_df = pd.DataFrame(data=X_arr, columns=new_column_names)
        return X_df

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Dataframe is expected.")
        for col in self.columns:
            values = X[col].sort_values()
            step_size = int(round(len(values) / self.bin_count))
            curr_edge_idx = step_size
            edges = []
            while len(edges) < self.bin_count - 1:
                edges.append(values.iloc[curr_edge_idx])
                curr_edge_idx += step_size
            self.intervalEdges[col] = [-np.inf]
            self.intervalEdges[col].extend(edges)
            self.intervalEdges[col].append(np.inf)
        return self
