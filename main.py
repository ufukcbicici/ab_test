import numpy as np

from dataset import Dataset
from recommender import Recommender

if __name__ == "__main__":
    target_scaling = {("n13", "A"): "power",
                      ("n13", "B"): "power",
                      ("n14", "A"): "log",
                      ("n14", "B"): "identity"}
    dataset = Dataset("toydata_mltest.csv")
    # dataset.plot_categorical_variables()
    dataset.plot_numerical_variables()

    # dataset.data_exploration()
    # recommender = Recommender(dataset=dataset, target_scaling=target_scaling)
    # recommender.train_regressors()
    # # dataset.data_exploration()
    # # dataset.prepare_dataset_v2()
    # # dataset.data_exploration()
