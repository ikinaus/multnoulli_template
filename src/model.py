import matplotlib.pyplot as plt
import numpy as np

from src.utils import softmax


class Multinoulli:
    def __init__(self, lr=0.1, max_iter=10000, min_step=1e-7, verbose=False):
        self.lr = lr
        self.max_iter = max_iter
        self.eps = min_step
        self.verbose = verbose
        self.losses = []

    def loss(self, Y, P):
        return -np.mean(np.sum(Y * np.log(P + 1e-15), axis=1))

    def loss_plot(self, losses):
        plt.plot(losses)
        plt.grid(True)
        plt.show()

    def fit(self, X, Y):

        self.theta = np.zeros((Y.shape[1], X.shape[1]))

        for _ in range(self.max_iter):
            old_theta = self.theta.copy()
            eta = X @ self.theta.T
            Phi = softmax(eta)
            grad = Y - Phi

            self.theta += (self.lr / X.shape[0]) * (grad.T @ X)
            self.losses.append(self.loss(Y, Phi))

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
