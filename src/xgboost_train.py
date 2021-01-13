
import pandas as pd
import numpy as np
from common_dataset import StockDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from matplotlib import font_manager
#from xgboost import plot_tree

# Calculate the total and yearly mean return every year
# by the Lump-Sum method
# Return (total return, yearly mean return)
def return_by_lump_sum(date, price):
  begin_date = date.min()
  end_date = date.max()
  begin_price = float(price[price['年月']==begin_date]['收盤價(元)_年'])
  end_price = float(price[price['年月']==end_date]['收盤價(元)_年'])

  total_return = (end_price-begin_price)/begin_price*100
  yearly_mean_return = 0
  if end_date-begin_date != 0:
    yearly_mean_return = ((end_price/begin_price)**(100/(end_date-begin_date))-1)*100

  return (total_return, yearly_mean_return)

# Calculate the total and yearly mean return every year
# by the Dollar-Cost-Averaging method
# Return (total return, yearly mean return)
def return_by_dca(date, price):
  begin_date = date.min()
  end_date = date.max()
  avg_price = float(np.mean(price['收盤價(元)_年']))
  end_price = float(price[price['年月']==end_date]['收盤價(元)_年'])

  total_return = (end_price-avg_price)/avg_price*100
  yearly_mean_return = 0
  # To avoid divide by zero
  if end_date-begin_date != 0:
    yearly_mean_return = ((end_price/avg_price)**(100/(end_date-begin_date))-1)*100

  return (total_return, yearly_mean_return)

# Training a classifier to classify stocks.
# and give the importance(information gain) of indexes
def select_index(train_d, val_d):
  # Prepare the dataset
  y_train = train_d['ReturnMean_year_Label']
  y_val = val_d['ReturnMean_year_Label']

  # Remap the labels
  y_train = y_train.map({-1:0, 1:1})
  y_val = y_val.map({-1:0, 1:1})

  excepted_col = [
    '證券代碼', # Misleading information
    '簡稱', # Misleading information
    '年月', # Useless inforamtion
    '市值(百萬元)', # 市值 = 股數*股價
    '收盤價(元)_年', # 市價無法反映基本面
    'Return', # Label
    'ReturnMean_year_Label' # Label
  ]
  x_train = train_d.drop(excepted_col, axis=1)
  x_val = val_d.drop(excepted_col, axis=1)

  d_train = xgb.DMatrix(x_train, label=y_train)
  d_val = xgb.DMatrix(x_val, label=y_val)

  # Training
  params_train = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.8,
    "base_score": np.mean(y_train),
    "eval_metric": "logloss",
    "max_depth":4,
  }
  model = xgb.train(
    params_train,
    d_train, 5000,
    evals = [(d_val, "Validation")],
    verbose_eval=100,
    early_stopping_rounds=20
  )

  return model

# Select best portfolio using the classfier above
def select_portfolio(model, train_d, val_d):
  # Filter out the first year of stocks in validation set
  first_year = val_d['年月'].min()
  first_year = val_d[val_d['年月']==first_year]

  # Prepare the testing dataset
  excepted_col = [
    '證券代碼', # Misleading information
    '簡稱', # Misleading information
    '年月', # Useless inforamtion
    '市值(百萬元)', # 市值 = 股數*股價
    '收盤價(元)_年', # 市價無法反映基本面
    'Return', # Label
    'ReturnMean_year_Label' # Label
  ]
  x_test = first_year.drop(excepted_col, axis=1)
  x_test = xgb.DMatrix(x_test)

  # Predict the class of stocks
  pred_label = model.predict(x_test)
  good_stocks = first_year[pred_label >= 0.5]['簡稱']
  print("Good stocks")
  print(list(good_stocks))

  portfolio = []
  for stock in good_stocks:
    history_d = train_d[train_d['簡稱']==stock]
    mean_PER = history_d['本益比'].mean()
    cur_PER = float(first_year[first_year['簡稱']==stock]['本益比'])

    # If current PE ratio lower than the margin of safety,
    # buy it.
    if (not np.isnan(mean_PER)) and cur_PER <= mean_PER*0.8:
      portfolio.append(stock)

  print("Good and cheap stocks")
  print(portfolio)

  return portfolio

def evaluate_portfolio(portfolio, val_d):
  avg_lumpsum_return = 0
  avg_lumpsum_yearly_mean_return = 0
  avg_dca_return = 0
  avg_dca_yearly_mean_return = 0

  print(f"Return of stocks in portfolio:")
  for stock_name in portfolio:
    stock_price = val_d[val_d['簡稱']==stock_name]
    val_date = stock_price['年月']

    # Calculate the profit by lump and DCA method
    lumpsum_return, lumpsum_yearly_mean_return = \
      return_by_lump_sum(val_date, stock_price)
    dca_return, dca_yearly_mean_return = \
      return_by_dca(val_date, stock_price)

    # Accumulate the average return of this TV
    avg_lumpsum_return += lumpsum_return
    avg_lumpsum_yearly_mean_return += lumpsum_yearly_mean_return
    avg_dca_return += dca_return
    avg_dca_yearly_mean_return += dca_yearly_mean_return

    # Print out the return information
    # print(f"{stock_name}-Lump-Sum:")
    # print(f"\tTotal Return: {lumpsum_return:.2f}%")
    # print(f"\tYearly Mean Return: {lumpsum_yearly_mean_return:.2f}%")
    # print(f"{stock_name}-DCA:")
    # print(f"\tTotal Return: {dca_return:.2f}%")
    # print(f"\tYearly Mean Return: {dca_yearly_mean_return:.2f}%")

  # Calculate and diaplay the average return of this TV
  avg_lumpsum_return /= len(portfolio)
  avg_lumpsum_yearly_mean_return /= len(portfolio)
  avg_dca_return /= len(portfolio)
  avg_dca_yearly_mean_return /= len(portfolio)

  return ( \
    avg_lumpsum_return, avg_lumpsum_yearly_mean_return, \
    avg_dca_return, avg_dca_yearly_mean_return \
  )

if __name__=="__main__":
  # Plot chinese
  plt.rcParams['font.family']=['Noto Sans CJK JP']
  plt.rcParams['font.sans-serif']=['Noto Sans CJK JP']

  # Load the dataset
  dataset = StockDataset()

  # Perform temperal validation
  date_list = dataset.filted_d['年月'].unique()
  training_mask = pd.Series(np.zeros((len(dataset.filted_d)), dtype=bool))

  total_lumpsum_return = []
  total_dca_return = []

  feature_importance = {}

  for tv,date in enumerate(date_list[:-1]):
    print(f"TV={tv}, Split={date}")
    
    # Split the training and validation data
    training_mask = training_mask | (dataset.filted_d['年月'] == date)
    train_d = dataset.filted_d.loc[training_mask]
    val_d = dataset.filted_d.loc[~training_mask]
    print(f"Train data: {len(train_d)}, Val data: {len(val_d)}")

    # Train the model to select index and portfolio
    print("Training...")
    model = select_index(train_d, val_d)
    portfolio = select_portfolio(model, train_d, val_d)

    print("Training was done.")

    # Plot the importance of arrtibute in the decision tree
    # ax = xgb.plot_importance(
    #   model,
    #   importance_type = "gain",
    #   title = f"基本面指標重要程度(TV={tv})",
    #   ylabel = "指標",
    #   show_values = False,
    #   grid = True
    # )
    # plt.show()

    tv_feature_importance = model.get_score(importance_type='gain')
    for feature, importance in tv_feature_importance.items():
      if feature not in feature_importance:
        feature_importance[feature]=importance
      else:
        feature_importance[feature]+=importance

    # Validate the return of each stock in the protfolio
    avg_lumpsum_return, \
    avg_lumpsum_yearly_mean_return, \
    avg_dca_return, \
    avg_dca_yearly_mean_return = evaluate_portfolio(portfolio, val_d)

    print("Lump-Sum return of this TV:")
    print(f"\tTotal Return: {avg_lumpsum_return:.2f}%")
    print(f"\tYearly Mean Return: {avg_lumpsum_yearly_mean_return:.2f}%")
    print(f"DCA return of this TV:")
    print(f"\tTotal Return: {avg_dca_return:.2f}%")
    print(f"\tYearly Mean Return: {avg_dca_yearly_mean_return:.2f}%")

    # Accumalate total yearly mean return of both Lump-Sum and DCA
    total_lumpsum_return.append(avg_lumpsum_yearly_mean_return)
    total_dca_return.append(avg_dca_yearly_mean_return)

    print("---")

  # Convert the totally return
  # Exclude the last TV
  total_lumpsum_return  = np.array(total_lumpsum_return[:-1])
  total_dca_return = np.array(total_dca_return[:-1])

  x_tv = list(range(len(date_list)-2))

  width = 0.25
  plt.bar(x_tv, total_lumpsum_return, label='Lump-Sum', width=0.25)
  plt.bar([p + width for p in x_tv], total_dca_return, label='DCA', width=0.25)
  plt.xticks([p + width/2 for p in x_tv], x_tv)
  plt.yticks(np.arange(0, 200, 1))
  plt.yscale('log')
  plt.legend()
  plt.title('Returns of XGBoost')
  plt.xlabel('Epochs of Temporal Validation')
  plt.ylabel('Yearly Mean Return(%)')  
  plt.show()

  # Draw feature importances
  features = []
  importances = []
  for feature, importance in feature_importance.items():
    importance /= len(date_list[:-1])
    features += [feature]
    importances += [importance]
    print(feature, importance)
  importances, features = zip(*sorted(zip(importances, features)))
  plt.barh(features, importances)
  plt.title('Feature Importance of XGBoost')
  plt.xlabel('F-measure')
  plt.ylabel('Feature')  
  plt.show()
