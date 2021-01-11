# Simple Moving average

import collections

class SMA:
  def __init__(self, size):
    self.__size = size
    self.__deque = collections.deque(maxlen=size)

  # Insert a new data to SMA
  def push(self, data):
    self.__deque.append(data)

  # Calculate the average value
  @property
  def mean(self):
    return sum(self.__deque)/len(self.__deque)

  # Calculate the slope across the window
  @property
  def slope(self):
    return (self.__deque[-1]-self.__deque[0])/len(self.__deque)