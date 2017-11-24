"""
Load this script to your python by running the following code
import requests
url = "https://raw.githubusercontent.com/abulbasar/machine-learning/master/Utils.py"
exec(requests.get(url).text)
   
"""
import math
import numpy as np
      
class Batchable:

    
    def shuffle(self):
        X = self.X
        y = self.y
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        self.X = X[idx, :]
        self.y = y[idx]
        self.require_shuffle = False
        return
    
    def __init__(self, X, y, batch_size = 128, seed = 1):
        np.random.seed(seed)
        self.X = X
        self.y = y
        self.shuffle()
        self.start = 0
        self.batch_size = batch_size
        self.num_batches = math.ceil(X.shape[0] / batch_size)
    
    def next(self):
        if self.require_shuffle:
            self.shuffle()
            
        start = self.start
        end = self.start + self.batch_size
        end = min(self.X.shape[0], end)
        if end == self.X.shape[0]:
            self.require_shuffle = True
        self.start = end % self.X.shape[0]
        return self.X[start: end, :], self.y[start: end]
    
def show(scores, ax = None):
    df = pd.DataFrame.from_dict(scores)
    if ax is None:
        _, ax = plt.subplots()
    df.plot.line(alpha = 0.4, ax = ax)
    df.rolling(100, min_periods=1).mean().plot.line(ax = ax)    

      
def plot_scores(scores, window = 10, plt = plt):
   """
   Parameters: 
   scores: dict containing iteration index as key and the cost value as value.
   window: length of the rolling window
   plt: matplotlib.pyplot module. Import it as plt if necessary.
   
   Output:
   Displays cost decay curve with rolling mean. 
   """
   s = pd.Series(scores)
   plt.plot(s, label = "original", alpha = 0.3, color = "steelblue")
   plt.plot(s.rolling(window).mean(), label = "rolling mean", color = "steelblue")
   plt.legend()
   plt.xlabel("Iterations")
   plt.ylabel("Cost")
   plt.title("Cost decay over iterations")
