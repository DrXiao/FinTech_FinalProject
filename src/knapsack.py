
import pandas as pd
import numpy as np
from common_dataset import StockDataset

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
  return ['台積電', '聯電', '聯發科', '鴻海']
  pass

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

    print("---")