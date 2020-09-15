import numpy as np

from data_reader import Dataset

if __name__ == "__main__":
    dataset = Dataset("toydata_mltest.csv", target_scaling={"n13": "power", "n14": "log"})
    # dataset.data_exploration()
    dataset.prepare_dataset_v2()
    # dataset.data_exploration()
