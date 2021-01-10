from pathlib import Path
import csv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler

data_path = Path('test/')


def read_dataset_from_csv(file):
    csv_file = open(data_path / file, "r", encoding="utf-8")
    dataset_rows = list(csv.reader(csv_file))
    csv_file.close()
    return dataset_rows


def standardlize_dataset(dataset):
    scaler = StandardScaler()
    scaler.fit(dataset_trainable)
    scaled_features = scaler.transform(dataset_trainable)
    dataset_features = pd.DataFrame(
        scaled_features, columns=dataset_trainable.columns)
    return dataset_features


if __name__ == "__main__":
    dataset_file = 'train.csv'
    dataset_rows = read_dataset_from_csv(dataset_file)

    dataset_train = pd.DataFrame(
        data=dataset_rows[1:], columns=dataset_rows[0])
    company_fields = ['證券代碼', '簡稱']

    # Discard fields to let dataset be trainable.
    dataset_trainable = dataset_train.drop(company_fields, axis=1)

    # dataset_features = standardlize_dataset(dataset_trainable)

    print(dataset_trainable)
