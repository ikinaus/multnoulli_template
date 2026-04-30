import matplotlib.pyplot as plt
import numpy as np

from src.utils import softmax


class Multinoulli:
    def __init__(self, lr=0.01, max_iter=10000, min_step=1e-5, verbose=False):
        self.lr = lr
        self.max_iter = max_iter
        self.eps = min_step
        self.verbose = verbose
        self.losses = []

    # def loss(self, y, eta):
    #     stable_log = np.maximum(0, eta) + np.log(1 + np.exp(-np.abs(eta)))
    #     return np.mean(-y * eta + stable_log)

    def loss_plot(self, losses):
        plt.plot(losses)
        plt.grid(True)
        plt.show()

    def fit(self, X, Y):

        self.theta = np.zeros((Y.shape[1], X.shape[1]))

        for _ in range(self.max_iter):
            old_theta = self.theta.copy()
            eta = X @ self.theta.T
            grad = Y - softmax(eta)

            self.theta += self.lr * (grad.T @ X)
            # self.losses.append(self.loss(Y, eta))

            if np.linalg.norm(self.theta - old_theta) < self.eps:
                break

        if self.verbose:
            self.loss_plot(self.losses)

    def predict_proba(self, X):
        eta = X @ self.theta.T
        return softmax(eta)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def get_weights(self):
        return self.theta

    def get_losses(self):
        return self.losses
