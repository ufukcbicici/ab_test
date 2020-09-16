from dataset import Dataset
from recommender_unified_groups import RecommenderUnifiedGroups

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--csv_path', required=True, type=str,
                        default="toydata_mltest.csv",
                        help='Enter the path of the .csv file')

    parser.add_argument('--lambda_', required=True, type=float,
                        default='0.5',
                        help='Weight score for ordering')

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    target_scaling = {"n13": "power", "n14": "log"}
    dataset = Dataset("toydata_mltest.csv")
    recommender = RecommenderUnifiedGroups(dataset=dataset, target_scaling=target_scaling)
    recommender.load_models()

    opts = parse_args()
    csv_path = opts.csv_path
    lambda_ = opts.lambda_
    assert 0.0 <= lambda_ <= 1.0
    recommender.score_csv_file(csv_path, lambda_=lambda_)

    # recommender.score_csv_file("toydata_mltest.csv", lambda_=0.5)

