class Batchable:

    def __init__(self, X, y, batch_size = 256, seed = None):
        import math
        import numpy as np
        if seed:
            np.random.seed(seed)
        idx = np.arange(X.shape[1])
        np.random.shuffle(idx)
        self.X = X[:, idx]
        self.y = y[:, idx]
        self.start = 0
        self.batch_size = batch_size
        self.num_batches = math.ceil(X.shape[0] / batch_size)
    
    def next(self):
        end = self.start + self.batch_size
        if end > self.X.shape[1]:
            end = self.X.shape[1] - 1
        return self.X[:, self.start: (end + 1)], self.y[:, self.start: (end + 1)]
