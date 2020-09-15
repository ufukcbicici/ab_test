import numpy as np

from data_reader import Dataset

if __name__ == "__main__":
    dataset = Dataset("toydata_mltest.csv", target_scaling={"n13": "identity", "n14": "identity"})
    # dataset.data_exploration()
    dataset.prepare_dataset_v2()
    # dataset.data_exploration()
