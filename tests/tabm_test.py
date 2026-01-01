import sys
print(sys.path)

import torch
import pytest

from tabm.tabm import (
    BatchEnsembleLinear,
    MiniEnsembleLinear,
    TabMBlock,
    TabM,
    TabMmini,
)


# =========================
# Fixtures
# =========================

@pytest.fixture
def batch_ensemble_linear():
    return BatchEnsembleLinear(
        in_features=8,
        out_features=4,
        ensemble_size=3,
        random_init=True,
    )


@pytest.fixture
def tabm_model():
    return TabM(
        input_dim=10,
        hidden_dims=[16, 16],
        output_dim=2,
        ensemble_size=4,
        dropout=0.1,
    )


@pytest.fixture
def tabmmini_model():
    return TabMmini(
        input_dim=10,
        hidden_dims=[16, 16],
        output_dim=2,
        ensemble_size=4,
        dropout=0.1,
    )


# =========================
# BatchEnsembleLinear tests
# =========================

def test_batchensemble_linear_shape(batch_ensemble_linear):
    x = torch.randn(5, 3, 8)
    y = batch_ensemble_linear(x)

    assert y.shape == (5, 3, 4)


def test_batchensemble_random_init_is_pm_one():
    layer = BatchEnsembleLinear(
        in_features=6,
        out_features=6,
        ensemble_size=5,
        random_init=True,
    )

    unique_R = torch.unique(layer.R)
    unique_S = torch.unique(layer.S)

    assert set(unique_R.tolist()).issubset({-1.0, 1.0})
    assert set(unique_S.tolist()).issubset({-1.0, 1.0})


def test_batchensemble_deterministic_init_is_one():
    layer = BatchEnsembleLinear(
        in_features=6,
        out_features=6,
        ensemble_size=5,
        random_init=False,
    )

    assert torch.all(layer.R == 1.0)
    assert torch.all(layer.S == 1.0)


# =========================
# MiniEnsembleLinear tests
# =========================

def test_miniensemble_linear_shape():
    layer = MiniEnsembleLinear(
        in_features=8,
        out_features=4,
        ensemble_size=3,
        random_init=True,
    )

    x = torch.randn(5, 3, 8)
    y = layer(x)

    assert y.shape == (5, 3, 4)


def test_miniensemble_random_init_is_pm_one():
    layer = MiniEnsembleLinear(
        in_features=6,
        out_features=6,
        ensemble_size=5,
        random_init=True,
    )

    unique_R = torch.unique(layer.R)
    assert set(unique_R.tolist()).issubset({-1.0, 1.0})


# =========================
# TabMBlock tests
# =========================

def test_tabm_block_shape():
    block = TabMBlock(
        in_features=8,
        out_features=12,
        ensemble_size=4,
        dropout=0.0,
        random_init=True,
    )

    x = torch.randn(7, 4, 8)
    y = block(x)

    assert y.shape == (7, 4, 12)


# =========================
# TabM tests
# =========================

def test_tabm_forward_shape(tabm_model):
    x = torch.randn(32, 10)
    y = tabm_model(x)

    assert y.shape == (32, 4, 2)


def test_only_first_block_has_random_adapters(tabm_model):
    first_block = tabm_model.blocks[0].linear
    later_block = tabm_model.blocks[1].linear

    assert not torch.all(first_block.R == 1.0)
    assert not torch.all(first_block.S == 1.0)

    assert torch.all(later_block.R == 1.0)
    assert torch.all(later_block.S == 1.0)


def test_tabm_backward_pass(tabm_model):
    x = torch.randn(16, 10)
    target = torch.randn(16, 4, 2)

    output = tabm_model(x)
    loss = torch.mean((output - target) ** 2)
    loss.backward()

    grads = [
        p.grad for p in tabm_model.parameters()
        if p.requires_grad
    ]

    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_tabm_determinism_with_seed():
    torch.manual_seed(0)
    model1 = TabM(10, [8], 1, ensemble_size=3)

    torch.manual_seed(0)
    model2 = TabM(10, [8], 1, ensemble_size=3)

    x = torch.randn(5, 10)

    y1 = model1(x)
    y2 = model2(x)

    assert torch.allclose(y1, y2)


# =========================
# TabMmini tests
# =========================

def test_tabmmini_forward_shape(tabmmini_model):
    x = torch.randn(32, 10)
    y = tabmmini_model(x)

    assert y.shape == (32, 4, 2)


def test_tabmmini_only_first_layer_has_adapter(tabmmini_model):
    # First layer must be MiniEnsembleLinear
    assert isinstance(tabmmini_model.first, MiniEnsembleLinear)

    # Later blocks must be fully shared
    for block in tabmmini_model.blocks:
        assert not hasattr(block.linear, "R")
        assert not hasattr(block.linear, "S")
        assert not hasattr(block.linear, "B")


def test_tabmmini_backward_pass(tabmmini_model):
    x = torch.randn(16, 10)
    target = torch.randn(16, 4, 2)

    output = tabmmini_model(x)
    loss = torch.mean((output - target) ** 2)
    loss.backward()

    grads = [
        p.grad for p in tabmmini_model.parameters()
        if p.requires_grad
    ]

    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_tabmmini_determinism_with_seed():
    torch.manual_seed(42)
    model1 = TabMmini(10, [8], 1, ensemble_size=3)

    torch.manual_seed(42)
    model2 = TabMmini(10, [8], 1, ensemble_size=3)

    x = torch.randn(5, 10)

    y1 = model1(x)
    y2 = model2(x)

    assert torch.allclose(y1, y2)
