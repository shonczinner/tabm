# Contains code to preprocess data as in TabM paper
# Noisy quantile transform / Piecewise Embedding

import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import check_random_state

# ------------------------------
# Noisy Quantile Transform
# ------------------------------

class NoisyQuantileTransformer:
    def __init__(self, n_quantiles=100, output_distribution='normal', noise=1e-3, random_state=None):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.noise = noise
        self.random_state = check_random_state(random_state)
        self.qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution)

    def fit(self, X):
        if self.noise > 0:
            X += self.random_state.normal(0.0, self.noise, X.shape)
        self.qt.fit(X)
        return self

    def transform(self, X):
        X_q = self.qt.transform(X)
        return X_q

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# ------------------------------
# Piecewise Linear Encoding
# ------------------------------

def ple_encode(x, bin_edges):
    x = np.asarray(x)
    T = len(bin_edges) - 1
    encoded = np.zeros((x.shape[0], T), dtype=np.float32)

    for t in range(T):
        b_left = bin_edges[t]
        b_right = bin_edges[t + 1]

        if t > 0:
            encoded[x < b_left, t] = 0.0
        if t < T - 1:
            encoded[x >= b_right, t] = 1.0
        mask = (x >= b_left) & (x < b_right)
        encoded[mask, t] = (x[mask] - b_left) / (b_right - b_left + 1e-12)

    # Handle extremes
    encoded[x >= bin_edges[-1], -1] = 1.0
    encoded[x < bin_edges[0], 0] = 0.0
    return encoded

# ------------------------------
# Combined Preprocessor
# ------------------------------

class Preprocessor:
    """
    Fits a NoisyQuantileTransformer and PLE on training data.
    Returns standardized numeric features and piecewise linear embeddings.
    Optionally standardizes y.
    """
    def __init__(self, n_bins=4, n_quantiles=100, noise=1e-3, random_state=None, standardize_y=False):
        self.n_bins = n_bins
        self.quant_transformer = NoisyQuantileTransformer(
            n_quantiles=n_quantiles, noise=noise, random_state=random_state
        )
        self.bin_edges_list = []
        self.standardize_y = standardize_y
        self.y_mean = None
        self.y_std = None

    # ------------------------------
    # PLE
    # ------------------------------
    def _ple_transform(self, X_norm):
        ple_columns = [ple_encode(X_norm[:, i], self.bin_edges_list[i])
                       for i in range(X_norm.shape[1])]
        # return as 3D array: (batch, features, segments)
        return np.stack(ple_columns, axis=1).astype(np.float32)

    # ------------------------------
    # Y
    # ------------------------------
    def _standardize_y(self, y):
        if self.standardize_y:
            return (y - self.y_mean) / (self.y_std + 1e-12)
        return y

    # ------------------------------
    # Fit
    # ------------------------------
    def fit(self, X_train, y_train=None):
        # Standardize / normalize numeric features
        X_train_norm = self.quant_transformer.fit_transform(X_train)

        # Fit PLE bins
        self.bin_edges_list = []
        for i in range(X_train_norm.shape[1]):
            col = X_train_norm[:, i]
            bin_edges = np.quantile(col, np.linspace(0, 1, self.n_bins + 1))
            self.bin_edges_list.append(bin_edges)

        # Compute PLE embeddings
        X_train_ple = self._ple_transform(X_train_norm)
        x_num_std = X_train_norm.astype(np.float32)

        # Fit y if requested
        if y_train is not None and self.standardize_y:
            y_train = np.asarray(y_train)
            self.y_mean = y_train.mean(axis=0)
            self.y_std = y_train.std(axis=0)
            y_train_std = self._standardize_y(y_train)
            return x_num_std, X_train_ple, y_train_std

        return x_num_std, X_train_ple

    # ------------------------------
    # Transform
    # ------------------------------
    def transform(self, X, y=None):
        X_norm = self.quant_transformer.transform(X)
        X_ple = self._ple_transform(X_norm)
        x_num_std = X_norm.astype(np.float32)

        if y is not None and self.standardize_y:
            y_std = self._standardize_y(np.asarray(y))
            return x_num_std, X_ple, y_std

        return x_num_std, X_ple

    # ------------------------------
    # Fit + Transform
    # ------------------------------
    def fit_transform(self, X_train, y_train=None):
        return self.fit(X_train, y_train)

