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

      
class Batchable(object):
    """
    Create a batchable object that would return batched X and y values.
    
    Usage:
    ....
    init = tf.global_variables_initializer()
    batchable = Batchable(X_train, y_train)
    with tf.Session() as sess:
    init.run()
    for i, progress, X_batch, y_batch in batchable.next():
        sess.run(opt, feed_dict={X: X_batch, y: y_batch})
        if i % (bachable.max_iters // 20) == 0:
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print("Progress:%3d%%" % progress, 
                  "Train accuracy: %.4f" % acc_train, 
                  "Test accuracy: %.4f" % acc_test)
    
    
    """
    
    def __shuffle(self):
        X = self.X
        y = self.y
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        self.X = X[idx]
        self.y = y[idx]
        self.require_shuffle = False
        return
    
    def __init__(self, X, y, batch_size = 32, epochs = 10, seed = 1):
        
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError('Both X and y must be np.ndarray')
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must be of same size of axis=0")
        
        from math import ceil
        np.random.seed(seed)
        self.X = X
        self.y = y
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches = ceil(X.shape[0] / batch_size)
        self.max_iters = self.epochs * self.num_batches
        self.__shuffle()
        
    def next(self):
        iteration, progress = 0, 0
        for i in range(self.epochs):
            self.current_epoch = i
            for j in range(self.num_batches):
                self.current_batch = j
                start = j * self.batch_size
                end = start + self.batch_size
                iteration = iteration + 1
                progress = int(100 * iteration / self.max_iters) + 1
                yield iteration, progress, self.X[start:end], self.y[start:end]
            self.__shuffle()


def data_generator(X, y, batch_size = 32, epochs = 1):
    from collections import namedtuple
    from math import ceil
    Batch = namedtuple("batch", ["epoch", "global_step", "progress", "X_batch", "y_batch"])
    global_step = 0
    for epoch in range(epochs):
        m = X.shape[0]
        indices = np.arange(m)
        np.random.shuffle(indices)
        X = X[indices]
        y = None if y is None else y[indices]
        num_batches = ceil(m/batch_size)
        for j in range(num_batches):
            start = j * batch_size
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = None if y is None else y[start:end]
            progress = (j + 1) * 100 / num_batches
            yield Batch(epoch, global_step, progress, X_batch, y_batch)
            global_step = global_step + 1
            

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




def to_categorical(y):
    y = y.flatten()
    depth = len(np.unique(y))
    eye = np.depth(depth)
    return eye[y]
