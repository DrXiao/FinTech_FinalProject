from pathlib import Path
import csv

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_path = Path('test/')
dataset_file = 'train.csv'
dataset_test_file = 'test.csv'

# dataset_features = standardlize_dataset(dataset_trainable)


def dataset_company(dataset_train):
    Filter = (dataset_train["年月"] == 200412)
    curr_set = set(list(dataset_train[Filter]['簡稱']))
    selected_dataset = pd.DataFrame(columns=dataset_train.columns)
    for company in list(curr_set):
        Filter = (dataset_train["簡稱"] == company)
        selected_dataset = pd.concat([selected_dataset, dataset_train[Filter]])
    return (selected_dataset.sort_index())


def dataset_preprocessing(dataset_train):
    return dataset_company(dataset_train)


if __name__ == "__main__":
    dataset_train = pd.read_csv(data_path / dataset_file)
    dataset_train.index += 1

    dataset_train = dataset_preprocessing(dataset_train)
    dataset_train.index = range(1, len(dataset_train) + 1)
    no_usage_fields = ['證券代碼', '簡稱', '年月', 'Return', 'ReturnMean_year_Label']

    company_fields = ['簡稱', "年月"]
    # print(dataset_train)
    # Discard fields to let dataset be trainable.
    dataset_info = dataset_train.drop(
        list(set(dataset_train.columns).difference(set(company_fields))), axis=1)
    dataset_trainable = dataset_train.drop(no_usage_fields, axis=1)
    dataset_label = dataset_train['ReturnMean_year_Label']

    train_data, test_data, train_label, test_label = \
        train_test_split(dataset_trainable, dataset_label,
                         test_size=0.5, shuffle=False)
    train_label = train_label.astype('int')
    test_label = test_label.astype('int')

    knn_obj = KNeighborsClassifier()

    knn_obj.fit(train_data, train_label)

    threshold = train_data['收盤價(元)_年'].sum() / len(train_data)
    # print(dataset_info)
    select_mode = 1 # int(input('1. Train Test,\n2. Formal test\t: '))
    if select_mode == 1:
        test_pred = knn_obj.predict(test_data)
        test_info_data = pd.merge(
            dataset_info, test_data, left_index=True, right_index=True)
        test_info_data['ReturnMean_year_Label'] = test_pred

    elif select_mode == 2:
        dataset_test = pd.read_csv(data_path / dataset_test_file)
        dataset_test_trainable = dataset_test.drop(no_usage_fields, axis=1)
        dataset_test_info = dataset_test.drop(
            list(set(dataset_test.columns).difference(set(company_fields))), axis=1)
        dataset_test_label = dataset_test['ReturnMean_year_Label']

        test_pred = knn_obj.predict(dataset_test_trainable)
        test_info_data = pd.merge(
            dataset_test_info, dataset_test_trainable, left_index=True, right_index=True)
        test_info_data['ReturnMean_year_Label'] = test_pred

    stocks_dict = {}
    profits = {}
    years = []

    start_year = list(test_info_data.head(1)["年月"])[0] // 100
    years_len = 1
    years.append(start_year)
    for _, row in test_info_data.iterrows():
        if(row["年月"] // 100 > start_year):
            start_year = row["年月"] // 100
            years.append(start_year)
            years_len += 1
        if row["ReturnMean_year_Label"] == -1:
            if row["簡稱"] in stocks_dict.keys() and len(stocks_dict[row["簡稱"]]) < 2 and (stocks_dict[row["簡稱"]][0] - row["收盤價(元)_年"]) / stocks_dict[row["簡稱"]][0] > 0.05:
                stocks_dict[row["簡稱"]].append(row["收盤價(元)_年"])
                profits[row["簡稱"]].append(row["收盤價(元)_年"])
        if row["收盤價(元)_年"] - threshold * 1.2:
            if row["簡稱"] in stocks_dict.keys():
                if len(stocks_dict[row["簡稱"]]) < 2 and (row["收盤價(元)_年"] - stocks_dict[row["簡稱"]][0]) / stocks_dict[row["簡稱"]][0] > 0.10:  # 賣
                    stocks_dict[row["簡稱"]].append(row["收盤價(元)_年"])
                    profits[row["簡稱"]].append(row["收盤價(元)_年"])
            elif (threshold * 1.2 - row["收盤價(元)_年"]) / (threshold * 1.2) > 0.05 and len(stocks_dict) < 4:
                stocks_dict[row["簡稱"]] = [row["收盤價(元)_年"]]   # 買
                profits[row["簡稱"]] = [row["收盤價(元)_年"]]
        if row["簡稱"] in stocks_dict.keys() and len(profits[row["簡稱"]]) < years_len:
            if len(stocks_dict[row["簡稱"]]) < 2:
                profits[row["簡稱"]].append(row["收盤價(元)_年"])
            else:
                profits[row["簡稱"]].append(profits[row["簡稱"]][-1])

    print(years)
    print(stocks_dict)
    print(profits)
    max_length = 0
    for key in profits:
        if len(profits[key]) > max_length:
            max_length = len(profits[key])
    for key in profits:
        if len(profits[key]) < max_length:
            profits[key].append(profits[key][-1])

    plt.subplots()
    dict_iter = iter(stocks_dict)
    for i in range(len(stocks_dict.keys())):
        plt.subplot(len(stocks_dict.keys()), 1, i + 1)
        plt.xlabel("year")
        plt.ylabel("price")
        plt.plot(years, profits[next(dict_iter)])
    plt.show()
    print('----------------------')

    for key in stocks_dict:
        if len(stocks_dict[key]) == 2:
            print(
                key, ":", 100*(stocks_dict[key][1] - stocks_dict[key][0]) / stocks_dict[key][0], "%")
        else:
            print(
                key, ":", 100*(profits[key][-1] - profits[key][0]) / profits[key][0], "%")    
