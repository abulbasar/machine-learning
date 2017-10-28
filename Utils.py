class Batchable:
    import math
    import numpy as np
    
    def shuffle(self):
        X = self.X
        y = self.y
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        self.X = X[idx, :]
        self.y = y[idx]
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
        start = self.start
        end = self.start + self.batch_size
        end = min(self.X.shape[0], end)
        self.start = end % self.X.shape[0]
        return self.X[start: end, :], self.y[start: end]
    
def show(scores, ax = None):
    df = pd.DataFrame.from_dict(scores)
    if ax is None:
        _, ax = plt.subplots()
    df.plot.line(alpha = 0.4, ax = ax)
    df.rolling(100, min_periods=1).mean().plot.line(ax = ax)    
