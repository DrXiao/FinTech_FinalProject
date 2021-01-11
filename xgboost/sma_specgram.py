import matplotlib .pyplot as plt
from pathlib import Path
import csv
from scipy.signal import stft
import numpy as np
import xgboost as xgb
from sma import SMA

from graphviz import Digraph
from xgboost import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



dataPath = Path("FinTech_FinalProject/xgboost/")

def sma_specgram(x, smas_size):

  smas = [SMA(i) for i in smas_size]

  sma_history = []
  slope_history = []

  for price in x:
    for sma in smas:
      sma.push(price)
    history_epoch = [sma.mean for sma in smas]
    slope_history_epoch = [sma.slope for sma in smas]
    sma_history.append(history_epoch)
    slope_history.append(slope_history_epoch)

  sma_history = np.array(sma_history)
  slope_history = np.array(slope_history)

  return sma_history, slope_history

def plot(y, smas_size, sma_history, slope_history, y_true, y_pred):
  #plt.rcParams['pcolor.shading'] = 'auto'

  sma_history = np.transpose(np.array(sma_history))
  slope_history = np.transpose(np.array(slope_history))

  fig, ax = plt.subplots(2, 2)
  ax = ax.flatten()

  # Draw the original stock price
  ax[0].set_title('Stock Price')
  ax[0].set(ylabel='Price', xlabel='Days')
  ax[0].plot(y)

  # Draw the predicted stock price
  ax[1].set_title('Predicted Stock Price')
  ax[1].set(ylabel='Price', xlabel='Days')
  ax[1].plot(y_true)
  ax[1].plot(y_pred)
  ax[1].legend(['True', 'Pred'], loc='upper left')

  # Draw the SMA of stock price
  ax[2].set_title('Simple Moving Average')
  ax[2].set(ylabel='Size of SMA', xlabel='Days')
  sma_color_map = ax[2].pcolormesh(range(len(y)), smas_size, sma_history)
  fig.colorbar(sma_color_map, ax=ax[2])

  # Draw the SMS of stock price
  ax[3].set_title('Simple Moving Slope')
  ax[3].set(ylabel='Size of SMS', xlabel='Days')
  slope_color_map = ax[3].pcolormesh(
    range(len(y)),
    smas_size,
    slope_history,
    cmap='coolwarm'
  )
  fig.colorbar(slope_color_map, ax=ax[3])
  mng = plt.get_current_fig_manager()
  mng.window.showMaximized()
  plt.show()


if __name__ == "__main__":

  dataset_filename = 'top200.csv'

  print("Dataset: {}".format(dataset_filename))

  csvFile = open(dataPath / dataset_filename, "r")

  rows = csv.reader(csvFile)

  rows = list(rows)[1:]

  raw_x=[]

  raw_y = [float(row[20]) for row in rows]

  # Initialize the window size of smas
  window_size = 1000
  step_size = 10
  smas_size = [i for i in range(1, window_size, step_size)]

  # set raw_x [市值、收盤價...]
  for i in range(len(rows)):
    raw_x.append(rows[i][3:19])
  

  # Shift one place as label
  x = raw_x
  y = raw_y[0:]

  sma_history, slope_history = sma_specgram(raw_y, smas_size)

  print("SMA and SMS calculated. Preparing dataset...")

  # Prepare the dataset

  #x = np.concatenate((sma_history, slope_history), axis=1)

  val_sep = int(len(raw_y)*0.8)
  x_train = np.array(x[:val_sep])
  x_val = np.array(x[val_sep:])
  y_train = np.array(y[:val_sep])
  y_val = np.array(y[val_sep:])

  

  # Training
  print("Begin training")

  xgb_r = xgb.XGBRegressor(
    learning_rate = 0.1,
    n_estimators = 10,
    max_depth = 8,
    subsample=0.8,
  ) 
  xgb_r.fit(x_train, y_train)

  # Predicting
  print("Training done. Predicting...")

  #print tree
  plot_tree(xgb_r, num_trees = 1)
  plt.show()
  plt.savefig('Tree from Top to Bottom1.png')

  # Plot the graph
  #plot(raw_y, smas_size, sma_history, slope_history, y_val, y_pred)

  # Simulate the investment

  y_pred = xgb_r.predict(x_val)
  bias = y_train[-1]/y_pred[-1]
  x=np.array(x)
  y_pred = xgb_r.predict(x)*bias

  print("y_pred top 5:")
  stock=[0]*5
  y_pred.sort()
  y_pred=y_pred[::-1]
  for i in range(5):
    stock[i]=y_pred[i]

  print(stock)


