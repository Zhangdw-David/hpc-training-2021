import numpy as np


class Lasso:
    """A simple Lasso regression implementation using proximal gradient descent."""

    def __init__(self, alpha=1.0, learning_rate=0.01, max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def _soft_threshold(value, threshold):
        return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n_samples, n_features = X.shape

        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean

        self.coef_ = np.zeros(n_features, dtype=float)

        for _ in range(self.max_iter):
            residual = X_centered @ self.coef_ - y_centered
            gradient = (X_centered.T @ residual) / n_samples
            new_coef = self._soft_threshold(
                self.coef_ - self.learning_rate * gradient,
                self.alpha * self.learning_rate,
            )

            if np.max(np.abs(new_coef - self.coef_)) < self.tol:
                self.coef_ = new_coef
                break

            self.coef_ = new_coef

        self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_
