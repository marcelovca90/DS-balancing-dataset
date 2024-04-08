import json
import os
import sys

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import LabelPowerset

from data_balancing.autoML_frameworks.utils import SET_SEED, GET_SEED
from optuna_search_tpe import optuna_search_tpe
from optuna_search_grid import optuna_search_grid

DATASET_PATH = "./artifacts/autobalancer_datasets/"
RESULTS_PATH = "./artifacts/autobalancer_optuna_results/"

def _train_test_split(X, y, test_size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=GET_SEED()
    )
    return X_train, X_test, y_train, y_test


def run_pipeline(dataset_id: int, framework_name: str):

    # =========================================================================
    # SPLIT TRAIN TEST
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    dataset_name, dataset_target = f'openml_{dataset_id}', 'class'
    full_csv_filename = os.path.join(DATASET_PATH, dataset_name + '.csv')
    train_csv_filename = full_csv_filename.replace('.csv', '_train.csv')
    test_csv_filename = full_csv_filename.replace('.csv', '_test.csv')

    if os.path.exists(train_csv_filename) and os.path.exists(test_csv_filename):
        train_data = pd.read_csv(train_csv_filename)
        test_data = pd.read_csv(test_csv_filename)
        X_train = train_data.drop(columns=[dataset_target])
        y_train = train_data[dataset_target]
        X_test = test_data.drop(columns=[dataset_target])
        y_test = test_data[dataset_target]

    else:
        dataset = fetch_openml(data_id=dataset_id, return_X_y=False)
        X, y = dataset.data.copy(deep=True), dataset.target.copy(deep=True)
        if dataset_id in [41465, 41468, 41470, 41471, 41473]:
            for col in y.columns.values:
                y[col] = y[col].map({'FALSE': 0, 'TRUE': 1}).to_numpy()
            y = pd.Series(LabelPowerset().transform(y))
        else:
            for col in X.columns.values:
                if X[col].dtype.name == 'category':
                    X.loc[:, col] = pd.Series(pd.factorize(X[col])[0])
            y = pd.Series(pd.factorize(y)[0])
        
        X_train, X_test, y_train, y_test = _train_test_split(X, y)

        full_data = X.assign(**{dataset_target: y})
        train_data = X_train.assign(**{dataset_target: y_train})
        test_data = X_test.assign(**{dataset_target: y_test})

        full_data.to_csv(full_csv_filename, index=False)
        train_data.to_csv(train_csv_filename, index=False)
        test_data.to_csv(test_csv_filename, index=False)

    # =========================================================================
    # APPLYING OPTUNA

    results = optuna_search_grid(
        train_dataset=train_data,
        test_dataset=test_data,
        target=dataset_target,
        dataset_name=dataset_name,
        framework_name=framework_name
    )

    # =========================================================================
    # PERSISTING RESULTS
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    results_filename = os.path.join(RESULTS_PATH, f'{dataset_name}_{framework_name}_results.csv')
    results.to_csv(results_filename)

if __name__ == "__main__":
    # =========================================================================
    # Reading shell variables

    print(f'sys.argv', sys.argv)
    if len(sys.argv) not in [3, 4]:
        print('Number of arguments must be 3 or 4: dataset_id framework_name [seed]')
    else:
        dataset_id = int(sys.argv[1])
        framework_name = str(sys.argv[2])
        if len(sys.argv) == 4:
            seed = int(sys.argv[3])
            SET_SEED(seed=seed)
        
    # SET_SEED(seed=41); run_pipeline(44, "autogluon")
    run_pipeline(dataset_id, framework_name)
