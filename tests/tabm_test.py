import torch
import pytest

from tabm.tabm import (
    BatchEnsembleLinear,
    MiniEnsembleLinear,
    TabMBlock,
    TabM,
    TabMMini,
)

# =========================
# Fixtures
# =========================

@pytest.fixture
def x_num():
    torch.manual_seed(42)
    return torch.randn(32, 10)

@pytest.fixture
def x_ple(x_num):
    batch_size, n_num = x_num.shape
    n_bins = 4
    torch.manual_seed(42)
    return torch.rand(batch_size, n_num, n_bins)

@pytest.fixture
def x_cat():
    torch.manual_seed(42)
    return torch.randint(0, 5, (32, 3))

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
        n_num=10,
        n_bins=4,
        n_cat=3,
        cat_cardinalities=[5, 5, 5],
        d_embedding=5,
        d_cat=4,
        hidden_dims=[16, 16],
        output_dim=2,
        ensemble_size=4,
        dropout=0.1,
        use_ple=True,
    )

@pytest.fixture
def tabm_model_numeric_only():
    return TabM(
        n_num=10,
        n_bins=None,
        n_cat=0,
        cat_cardinalities=None,
        d_embedding=5,
        d_cat=0,
        hidden_dims=[16, 16],
        output_dim=2,
        ensemble_size=4,
        dropout=0.1,
        use_ple=False,
    )

@pytest.fixture
def tabm_model_cat_only():
    return TabM(
        n_num=0,
        n_bins=None,
        n_cat=3,
        cat_cardinalities=[5, 5, 5],
        d_embedding=0,
        d_cat=4,
        hidden_dims=[16],
        output_dim=2,
        ensemble_size=4,
        dropout=0.1,
        use_ple=False,
    )

@pytest.fixture
def tabmmini_model():
    return TabMMini(
        n_num=10,
        n_bins=4,
        n_cat=3,
        cat_cardinalities=[5, 5, 5],
        d_embedding=5,
        d_cat=4,
        hidden_dims=[16, 16],
        output_dim=2,
        ensemble_size=4,
        dropout=0.1,
        use_ple=True,
    )

# =========================
# BatchEnsembleLinear tests
# =========================

def test_batchensemble_linear_shape(batch_ensemble_linear):
    x = torch.randn(5, 3, 8)
    y = batch_ensemble_linear(x)
    assert y.shape == (5, 3, 4)

def test_batchensemble_random_init_is_pm_one():
    layer = BatchEnsembleLinear(6, 6, 5, random_init=True)
    assert set(torch.unique(layer.R).tolist()).issubset({-1.0, 1.0})
    assert set(torch.unique(layer.S).tolist()).issubset({-1.0, 1.0})

def test_batchensemble_deterministic_init_is_one():
    layer = BatchEnsembleLinear(6, 6, 5, random_init=False)
    assert torch.all(layer.R == 1.0)
    assert torch.all(layer.S == 1.0)

# =========================
# MiniEnsembleLinear tests
# =========================

def test_miniensemble_linear_shape():
    layer = MiniEnsembleLinear(8, 4, 3)
    x = torch.randn(5, 3, 8)
    y = layer(x)
    assert y.shape == (5, 3, 4)

def test_miniensemble_R_is_pm_one():
    layer = MiniEnsembleLinear(6, 6, 5)
    assert set(torch.unique(layer.R).tolist()).issubset({-1.0, 1.0})

# =========================
# TabMBlock tests
# =========================

def test_tabm_block_shape():
    block = TabMBlock(8, 12, 4, dropout=0.0, random_init=True)
    x = torch.randn(7, 4, 8)
    y = block(x)
    assert y.shape == (7, 4, 12)

# =========================
# TabM tests
# =========================

def test_tabm_forward_shape(tabm_model, x_num, x_ple, x_cat):
    y = tabm_model(x_num=x_num, x_ple=x_ple, x_cat=x_cat)
    assert y.shape == (32, 4, 2)

def test_tabm_numeric_only(tabm_model_numeric_only, x_num):
    y = tabm_model_numeric_only(x_num=x_num)
    assert y.shape == (32, 4, 2)

def test_tabm_cat_only(tabm_model_cat_only, x_cat):
    y = tabm_model_cat_only(x_cat=x_cat)
    assert y.shape == (32, 4, 2)

def test_tabm_requires_ple_enforced(tabm_model, x_num):
    with pytest.raises(AssertionError):
        tabm_model(x_num=x_num, x_ple=None)

def test_only_first_block_has_random_adapters(tabm_model):
    first = tabm_model.blocks[0].linear
    later = tabm_model.blocks[1].linear
    assert not torch.all(first.R == 1.0)
    assert not torch.all(first.S == 1.0)
    assert torch.all(later.R == 1.0)
    assert torch.all(later.S == 1.0)

def test_tabm_backward_pass(tabm_model, x_num, x_ple, x_cat):
    target = torch.randn(32, 4, 2)
    output = tabm_model(x_num=x_num, x_ple=x_ple, x_cat=x_cat)
    loss = torch.mean((output - target) ** 2)
    loss.backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in tabm_model.parameters()
        if p.requires_grad
    )

# =========================
# TabMMini tests
# =========================

def test_tabmmini_forward_shape(tabmmini_model, x_num, x_ple, x_cat):
    y = tabmmini_model(x_num=x_num, x_ple=x_ple, x_cat=x_cat)
    assert y.shape == (32, 4, 2)

def test_tabmmini_backward_pass(tabmmini_model, x_num, x_ple, x_cat):
    target = torch.randn(32, 4, 2)
    output = tabmmini_model(x_num=x_num, x_ple=x_ple, x_cat=x_cat)
    loss = torch.mean((output - target) ** 2)
    loss.backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in tabmmini_model.parameters()
        if p.requires_grad
    )
