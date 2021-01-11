
import pandas as pd
import numpy as np
from common_dataset import StockDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

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

# Select best portfolio using risk and potential return via dynamic programming
def select_portfolio(train_d, excepted_risk=0.1):
  stocks = np.array(train_d['簡稱'].unique())
  dp_item = []
  for stock in stocks:
    # Estimate the return
    stock_d = train_d[train_d['簡稱']==stock]
    train_x = stock_d.drop
    excepted_fields = ['證券代碼', '簡稱', '市值(百萬元)']
    self.train_d = dataset.drop(excepted_fields, axis=1).reindex()
    train_y = train_y[train_d['簡稱']]
    clf = SVR(kernel='rbf', gamma=0.1)
    clf.fit(, y)

    # Calculate the risk
    price = train_d[train_d['簡稱']==stock]['收盤價(元)_年']
    mean_price = np.repeat(np.mean(price), len(price))
    rmse = mean_squared_error(price, mean_price, squared = False)
    dp_item.append({stock:stock, risk:rmse})
    print(f"Risk of {stock}: {rmse}")

  # Select the portfolio via dynamic programming


  return ['台積電', '聯電', '聯發科', '鴻海']

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
    print(f"{stock_name}-Lump-Sum:")
    print(f"\tTotal Return: {lumpsum_return:.2f}%")
    print(f"\tYearly Mean Return: {lumpsum_yearly_mean_return:.2f}%")
    print(f"{stock_name}-DCA:")
    print(f"\tTotal Return: {dca_return:.2f}%")
    print(f"\tYearly Mean Return: {dca_yearly_mean_return:.2f}%")

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

  # Load the dataset
  dataset = StockDataset()

  # Perform temperal validation
  date_list = dataset.filted_d['年月'].unique()
  training_mask = pd.Series(np.zeros((len(dataset.filted_d)), dtype=bool))

  total_lumpsum_return = []
  total_dca_return = []

  for tv,date in enumerate(date_list[:-1]):
    print(f"TV={tv}, Split={date}")
    
    # Split the training and validation data
    training_mask = training_mask | (dataset.filted_d['年月'] == date)
    train_d = dataset.filted_d.loc[training_mask]
    val_d = dataset.filted_d.loc[~training_mask]
    print(f"Train data: {len(train_d)}, Val data: {len(val_d)}")

    # Train the model to select portfolio
    print("Training...")
    portfolio = select_portfolio(train_d)

    print("Training was done.")

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
  plt.title('Returns of 0/1 Knapsack')
  plt.xlabel('Epochs of Temporal Validation')
  plt.ylabel('Yearly Mean Return(%)')  
  plt.show()
