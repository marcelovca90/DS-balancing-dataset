import os
import pandas as pd

from sklearn.datasets import fetch_openml
from skmultilearn.problem_transform import LabelPowerset

def detect_nas():
    base_folder = 'artifacts/autobalancer_datasets'
    all_files = os.listdir(base_folder)
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    for filename in csv_files:
        df = pd.read_csv(os.path.join(base_folder, filename))
        print(filename, df.isnull().sum()[df.isnull().sum() > 0])

def apply_multilabel_powerset(dataset_id):
    dataset = fetch_openml(data_id=dataset_id, return_X_y=False)
    X, y = dataset.data.copy(deep=True), dataset.target.copy(deep=True)
    for col in y.columns.values:
        y[col] = y[col].map({'FALSE': 0, 'TRUE': 1}).to_numpy()
    y = pd.Series(LabelPowerset().transform(y))
    df = pd.DataFrame(X).assign(**{'class_ps': pd.Series(y)})
    return df

if __name__ == "__main__":
    for dataset_id in [41465, 41468, 41470, 41471, 41473]:
        df = apply_multilabel_powerset(dataset_id)
        filename = os.path.join('artifacts/autobalancer_datasets', f'openml_{dataset_id}.csv')
        df.to_csv(filename, index=False)