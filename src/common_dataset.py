from pathlib import Path
import csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_path = Path('test/')
default_dataset_file = data_path / 'train.csv'

# Read the stock dataset from csv
class StockDataset:
  def __init__(self, dataset_file=default_dataset_file):
    print(f"Dataset: {dataset_file}")

    # Preprocessing
    dataset = pd.read_csv(dataset_file, encoding="utf-8")
    self.raw_d = dataset

    max_date = dataset['年月'].max()
    len_yaers = len(dataset.groupby('年月'))
    self.max_date = max_date
    self.len_yaers = len_yaers
    self.dates = dataset['年月']

    print(f"Maximum date of dataset: {max_date}")
    print(f"Length of years: {len_yaers}")

    # Filter out the stocks that delisted during the dataset
    top_stocks_filter = dataset[dataset['年月']==max_date]['簡稱'].unique()
    dataset = dataset[dataset['簡稱'].isin(top_stocks_filter)]
    self.stocks = top_stocks_filter

    self.filted_d = dataset.reindex()
    print(f"Filtered data: {len(dataset)}")

    # Discard fields to let dataset be trainable.
    excepted_fields = ['證券代碼', '簡稱', '市值(百萬元)']
    self.train_d = dataset.drop(excepted_fields, axis=1).reindex()

    print("================")

# For testing
if __name__=="__main__":
  dataset = StockDataset()
  print(dataset.train_d)