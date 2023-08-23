import numpy as np


class Model:
    def __init__(self, thetas, alpha=0.01, max_iter=1000, eps=1e-15, batch=None, log=False):
        self.thetas = thetas
        self.alpha = alpha
        self.max_iter = max_iter
        self.eps = eps
        self.batch = batch
        self.log = log

        if self.log:
            self.loss_hist = []

    def predict(self, x):
        if len(x) == 0:
            return None

        try:
            if self.thetas.shape[1] != 1:
                return None

            if x.shape[1] + 1 != self.thetas.shape[0]:
                return None

            ones = np.ones((x.shape[0], 1))
            x_prime = np.concatenate((ones, x), axis=1)
            y_hat = x_prime.dot(self.thetas)

            return 1 / (1 + np.exp(-y_hat))
        except:
            return None


    def loss(self, y, y_hat):
        try:
            if y.shape != y_hat.shape:
                return None

            if y.shape[1] != 1:
                return None

            if len(y) == 0:
                return None

            total = y * np.log(y_hat + self.eps) + (1 - y) * np.log((1 - y_hat) + self.eps)
            return -total.sum() / len(y)
        except:
            return None


    def fit(self, x, y):
        try:
            if y.shape[1] != 1 or self.thetas.shape[1] != 1:
                return None

            if x.shape[0] != y.shape[0]:
                return None

            if x.shape[1] + 1 != self.thetas.shape[0]:
                return None

            ones = np.ones((x.shape[0], 1))
            x_prime = np.concatenate((ones, x), axis=1)
            new_thetas = np.copy(self.thetas)

            if self.batch is not None:
                total = np.concatenate((x_prime, y), axis=1)

            for _ in range(self.max_iter):
                if self.batch is None:
                    batch_x, batch_y = x_prime, y
                else:
                    perm = np.random.permutation(total)
                    batch_x, batch_y = perm[:self.batch + 1, :-1], perm[:self.batch + 1, -1].reshape(-1, 1)

                predictions = batch_x.dot(new_thetas).astype(float)
                predictions = 1 / (1 + np.exp(-predictions))

                gradient = batch_x.T.dot(predictions - batch_y)
                gradient /= len(batch_y)

                if self.log:
                    self.loss_hist.append(self.loss(batch_y, predictions))

                new_thetas -= (gradient * self.alpha)
            self.thetas = new_thetas

        except Exception as e:
            print(e)
            return None


    @staticmethod
    def accuracy(y, y_hat):
        try:
            matches = (y == y_hat)
            return np.count_nonzero(matches) / len(matches)
        except:
            return None
