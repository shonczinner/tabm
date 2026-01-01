import torch
import pytest
from tabm import losses

# ------------------------------
# Fixtures
# ------------------------------

@pytest.fixture
def dummy_regression_data():
    batch, ensemble, out_dim = 5, 3, 2
    y_pred = torch.randn(batch, ensemble, out_dim)
    y_true = torch.randn(batch, out_dim)
    return y_pred, y_true

@pytest.fixture
def dummy_classification_data():
    batch, ensemble, num_classes = 6, 4, 3
    y_pred = torch.randn(batch, ensemble, num_classes)  # logits
    y_true = torch.randint(0, num_classes, (batch,))
    return y_pred, y_true

# ------------------------------
# Regression Loss Tests
# ------------------------------

def test_regression_loss_shape_training(dummy_regression_data):
    y_pred, y_true = dummy_regression_data
    loss = losses.regression_loss(y_pred, y_true, training=True, reduction='none')
    assert loss.shape == (y_pred.shape[0], y_pred.shape[1])  # (batch, ensemble)

def test_regression_loss_shape_validation(dummy_regression_data):
    y_pred, y_true = dummy_regression_data
    loss = losses.regression_loss(y_pred, y_true, training=False, reduction='none')
    assert loss.shape[0] == y_pred.shape[0]  # (batch,) averaged across ensemble

def test_regression_loss_mean(dummy_regression_data):
    y_pred, y_true = dummy_regression_data
    loss = losses.regression_loss(y_pred, y_true, training=True, reduction='mean')
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0  # scalar


# ------------------------------
# Classification Loss Tests
# ------------------------------

def test_classification_loss_training_shape(dummy_classification_data):
    y_pred, y_true = dummy_classification_data
    loss = losses.classification_loss(y_pred, y_true, training=True, reduction='none')
    assert loss.shape == (y_pred.shape[0], y_pred.shape[1])  # (batch, ensemble)

def test_classification_loss_training_mean(dummy_classification_data):
    y_pred, y_true = dummy_classification_data
    loss = losses.classification_loss(y_pred, y_true, training=True, reduction='mean')
    assert loss.ndim == 0  # scalar

def test_classification_loss_validation_shape(dummy_classification_data):
    y_pred, y_true = dummy_classification_data
    loss = losses.classification_loss(y_pred, y_true, training=False, reduction='none')
    assert loss.shape[0] == y_pred.shape[0]  # (batch,)

def test_classification_loss_validation_mean(dummy_classification_data):
    y_pred, y_true = dummy_classification_data
    loss = losses.classification_loss(y_pred, y_true, training=False, reduction='mean')
    assert loss.ndim == 0  # scalar

def test_classification_loss_avg_vs_individual(dummy_classification_data):
    y_pred, y_true = dummy_classification_data
    loss_train = losses.classification_loss(y_pred, y_true, training=True, reduction='none')  # (batch, ensemble)
    loss_val = losses.classification_loss(y_pred, y_true, training=False, reduction='none')   # (batch,)
    
    # 1. Ensure shapes are correct
    assert loss_val.shape[0] == loss_train.shape[0]
    
    # 2. Ensure values are finite and non-negative
    assert torch.all(loss_val >= 0.0)
    assert torch.all(torch.isfinite(loss_val))
