import numpy as np
import pytest
from tabm import preprocess

# ------------------------------
# Fixtures
# ------------------------------

@pytest.fixture
def dummy_numeric_data():
    np.random.seed(42)
    X = np.random.rand(50, 4) * 10  # 50 samples, 4 features
    return X

@pytest.fixture
def dummy_vector():
    np.random.seed(42)
    x = np.random.rand(20) * 5
    return x

@pytest.fixture
def dummy_target():
    np.random.seed(42)
    y = np.random.randn(50) * 5 + 10  # regression target
    return y

# ------------------------------
# Noisy Quantile + PLE Tests
# ------------------------------

def test_preprocessor_fit_transform_shape(dummy_numeric_data):
    pre = preprocess.Preprocessor(n_bins=4, noise=1e-4, random_state=42)
    x_num_std, X_ple = pre.fit_transform(dummy_numeric_data)
    # x_num_std shape: (batch, features)
    assert x_num_std.shape == dummy_numeric_data.shape
    # X_ple shape: (batch, features, n_bins)
    assert X_ple.shape == (dummy_numeric_data.shape[0], dummy_numeric_data.shape[1], 4)

def test_preprocessor_transform_consistency(dummy_numeric_data):
    pre = preprocess.Preprocessor(n_bins=4, noise=0, random_state=42)
    x_num_std_train, X_train_ple = pre.fit_transform(dummy_numeric_data)
    x_num_std_new, X_new_ple = pre.transform(dummy_numeric_data)
    # With noise=0, numeric features and PLE should match
    assert np.allclose(x_num_std_train, x_num_std_new)
    assert np.allclose(X_train_ple, X_new_ple)

def test_preprocessor_values_range(dummy_numeric_data):
    pre = preprocess.Preprocessor(n_bins=4, noise=0, random_state=42)
    _, X_ple = pre.fit_transform(dummy_numeric_data)
    # All values in [0,1]
    assert X_ple.min() >= 0.0 and X_ple.max() <= 1.0

def test_ple_encode_edges(dummy_vector):
    x = dummy_vector
    bins = np.quantile(x, np.linspace(0, 1, 5))  # 4 bins
    encoded = preprocess.ple_encode(x, bins)
    # First column should contain at least one 0
    assert np.any(encoded[:, 0] == 0.0)
    # Last column should contain at least one 1
    assert np.any(encoded[:, -1] == 1.0)
    # Values in [0,1]
    assert encoded.min() >= 0.0 and encoded.max() <= 1.0

# ------------------------------
# Optional y standardization tests
# ------------------------------

def test_preprocessor_with_y(dummy_numeric_data, dummy_target):
    pre = preprocess.Preprocessor(n_bins=4, standardize_y=True, noise=0, random_state=42)
    x_num_std, X_ple, y_std = pre.fit_transform(dummy_numeric_data, dummy_target)
    # x_num_std shape
    assert x_num_std.shape == dummy_numeric_data.shape
    # X_ple shape
    assert X_ple.shape == (dummy_numeric_data.shape[0], dummy_numeric_data.shape[1], 4)
    # y standardized: mean approx 0, std approx 1
    assert np.allclose(y_std.mean(), 0, atol=1e-6)
    assert np.allclose(y_std.std(), 1, atol=1e-6)

def test_preprocessor_transform_with_y(dummy_numeric_data, dummy_target):
    pre = preprocess.Preprocessor(n_bins=4, standardize_y=True, noise=0, random_state=42)
    x_num_std_train, X_train_ple, y_train_std = pre.fit_transform(dummy_numeric_data, dummy_target)
    x_num_std_new, X_new_ple, y_new_std = pre.transform(dummy_numeric_data, dummy_target)
    # With noise=0, numeric features and PLE should match
    assert np.allclose(x_num_std_train, x_num_std_new)
    assert np.allclose(X_train_ple, X_new_ple)
    # y standardized should match
    assert np.allclose(y_train_std, y_new_std)
