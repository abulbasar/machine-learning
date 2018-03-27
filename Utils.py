"""
Load this script to your python by running the following code
import requests
url = "https://raw.githubusercontent.com/abulbasar/machine-learning/master/Utils.py"
exec(requests.get(url).text)
   
"""
import math
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing

      
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
    
    def __init__(self, X, y, batch_size = 32, seed = 1):
        np.random.seed(seed)
        self.X = X
        self.y = y
        self.shuffle()
        self.start = 0
        self.batch_size = batch_size
        self.num_batches = math.ceil(X.shape[0] / batch_size)
    
    def next(self):
        start = self.start
        end = self.start + self.batch_size
        end = min(self.X.shape[0], end)
        self.start = end % self.X.shape[0]
        return self.X[start: end, :], self.y[start: end]
    

def plot_scores(scores, window = 10):

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
   
   
class CifarLoader(object):
    """
    Loads CIFAR10 dataset
    
    """
   
    def load_data(self, files):
        import pickle
        import numpy as np
        X = np.empty([0, 3072])
        y = np.empty([0])
        for path in files:
            print(path)
            with open(path, "rb") as f:
                d = pickle.load(f, encoding='bytes')
                X = np.vstack([X, d[b"data"]]).astype("uint8")
                y = np.hstack([y, d[b"labels"]]).astype("uint8")
        return X, y
        
    def __init__(self, data_path):
        import os, pickle
        
        training_files = [os.path.join(data_path, "data_batch_{0}".format(i))  for i in range(1, 6)]
        test_files = [os.path.join(data_path, "test_batch")]
        labels_file = os.path.join(data_path, "batches.meta")

        X_train, y_train = self.load_data(training_files)
        X_test, y_test = self.load_data(test_files)
        
        self.X_train = X_train.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])/255
        self.X_test = X_test.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])/255
        self.y_train = y_train
        self.y_test = y_test
        
        with open(labels_file, "rb") as f:
            labels = pickle.load(f, encoding="bytes")
        labels = [s.decode("utf-8")  for s in labels[b'label_names']]
        
        self.labels = labels
    
    def __repr__(self):
        row_format ="{:<15}" * 2
        lines = [
            row_format.format("X_train", str(self.X_train.shape)),
            row_format.format("X_test", str(self.X_test.shape)),
            row_format.format("y_train", str(self.y_train.shape)),
            row_format.format("y_test", str(self.y_test.shape)),
            row_format.format("labels", str(self.labels))
        ]
        return "\n".join(lines)

   
   
import numpy as np

def outliers(y):
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lower_bound = max(np.min(y), q1 - (iqr * 1.5))
    upper_bound = min(np.max(y), q3 + (iqr * 1.5))
    return (y > upper_bound) | (y < lower_bound)


def load_mnist_csv(path = "/data/MNIST/", one_hot = False, shape = None):
    df_train = pd.read_csv(path + "mnist_train.csv", header=None)
    df_test = pd.read_csv(path + "mnist_test.csv", header=None)
    
    X_train = df_train.iloc[:, 1:].values/255
    X_test = df_test.iloc[:, 1:].values/255
    y_train = df_train.iloc[:, 0].values
    y_test = df_test.iloc[:, 0].values
    
    if shape == "2D":
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
        
    if shape == "3D":
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
    
    if one_hot:
        eye = np.eye(len(np.unique(y_train)))
        y_train, y_test = eye[y_train], eye[y_test]
        
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_mnist_csv(shape = "2D")


def to_categorical(y):
    y = y.flatten()
    depth = len(np.unique(y))
    eye = np.depth(depth)
    return eye[y]
