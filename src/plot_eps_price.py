import pandas as pd
import numpy as np
from common_dataset import StockDataset
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
  # Plot chinese
  plt.rcParams['font.family']=['Noto Sans CJK JP']
  plt.rcParams['font.sans-serif']=['Noto Sans CJK JP']

  # Load the dataset
  dataset = StockDataset()

  stocks = dataset.filted_d['簡稱'].unique()

  for stock in stocks:
    d = dataset.filted_d[dataset.filted_d['簡稱']==stock]
    #d = dataset.filted_d
    price_cur_year = pd.to_numeric(d['收盤價(元)_年'].shift(-1), downcast='float')[:-1]
    price_last_year = pd.to_numeric(d['收盤價(元)_年'], downcast='float')[:-1]

    eps = d['收盤價(元)_年']/d['本益比']
    eps_cur_year = pd.to_numeric(eps.shift(-1), downcast='float')[:-1]
    eps_last_year = pd.to_numeric(eps, downcast='float')[:-1]

    # Calculate YoY
    x_eps_yoy = (eps_cur_year - eps_last_year)/eps_last_year*100
    y_price_yoy = (price_cur_year - price_last_year)/price_last_year*100

    # Standarlize
    x_eps_yoy = (x_eps_yoy - x_eps_yoy.mean())/(x_eps_yoy.std())
    y_price_yoy = (y_price_yoy - y_price_yoy.mean())/(y_price_yoy.std())


    # Calculate the regression line
    m, b = np.polyfit(x_eps_yoy, y_price_yoy, 1)
    mse = (np.square(y_price_yoy-(m*x_eps_yoy + b))).mean()

    plt.title(f"{stock} (R={m:.2}, MSE={mse:.2})")
    plt.scatter(x_eps_yoy, y_price_yoy)
    plt.plot(x_eps_yoy, m*x_eps_yoy + b, color='orange')
    plt.ylabel("Stock Price YoY")
    plt.xlabel("EPE YoY ")
    plt.show()