# Contains pytorch modules for tabm. TabM = MLP with BatchEnsemble + Better Initialization and TabMMini

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# =========================
# BatchEnsemble Linear
# =========================

class BatchEnsembleLinear(nn.Module):
    """
    BatchEnsemble linear layer:
    l_BE(X) = ((X ⊙ R) W) ⊙ S + B

    X: (batch, ensemble, in_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        bias: bool = True,
        random_init: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        # Shared weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        # Adapters
        self.R = nn.Parameter(torch.ones(ensemble_size, in_features))
        self.S = nn.Parameter(torch.ones(ensemble_size, out_features))

        if bias:
            self.B = nn.Parameter(torch.zeros(ensemble_size, out_features))
        else:
            self.register_parameter("B", None)

        # Better initialization (±1)
        if random_init:
            self.R.data = torch.randint(0, 2, self.R.shape, dtype=torch.float32) * 2 - 1
            self.S.data = torch.randint(0, 2, self.S.shape, dtype=torch.float32) * 2 - 1

    def forward(self, x):
        # x: (batch, ensemble, in_features)

        x = x * rearrange(self.R, "e d -> 1 e d")
        x = torch.matmul(x, rearrange(self.weight, "o i -> i o"))
        x = x * rearrange(self.S, "e d -> 1 e d")

        if self.B is not None:
            x = x + rearrange(self.B, "e d -> 1 e d")

        return x


# =========================
# MiniEnsemble Linear
# =========================

class MiniEnsembleLinear(nn.Module):
    """
    MiniEnsemble linear layer:
    l_ME(X) = (X ⊙ R) W

    Only the first adapter R is used.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        random_init: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        self.R = nn.Parameter(torch.ones(ensemble_size, in_features))

        if random_init:
            self.R.data = torch.randint(0, 2, self.R.shape, dtype=torch.float32) * 2 - 1

    def forward(self, x):
        # x: (batch, ensemble, in_features)
        x = x * rearrange(self.R, "e d -> 1 e d")
        x = torch.matmul(x, rearrange(self.weight, "o i -> i o"))
        return x


# =========================
# Blocks
# =========================

class TabMBlock(nn.Module):
    """
    BatchEnsemble block:
    Linear → ReLU → Dropout
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        dropout: float,
        random_init: bool = False,
    ):
        super().__init__()

        self.linear = BatchEnsembleLinear(
            in_features,
            out_features,
            ensemble_size,
            random_init=random_init,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class SharedBlock(nn.Module):
    """
    Fully shared block (no adapters):
    Linear → ReLU → Dropout
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, ensemble, features)
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


# =========================
# TabM
# =========================

class TabM(nn.Module):
    """
    TabM = MLP + BatchEnsemble + Better Initialization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        ensemble_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        dims = [input_dim] + hidden_dims
        self.blocks = nn.ModuleList()

        for i in range(len(hidden_dims)):
            self.blocks.append(
                TabMBlock(
                    dims[i],
                    dims[i + 1],
                    ensemble_size,
                    dropout,
                    random_init=(i == 0),
                )
            )

        self.head = BatchEnsembleLinear(
            hidden_dims[-1],
            output_dim,
            ensemble_size,
            random_init=False,
        )

    def forward(self, x):
        # x: (batch, input_dim)
        x = repeat(x, "b d -> b e d", e=self.head.ensemble_size)

        for block in self.blocks:
            x = block(x)

        return self.head(x)


# =========================
# TabMmini
# =========================

class TabMmini(nn.Module):
    """
    TabMmini = MLP + MiniEnsemble

    Only the very first adapter R is used.
    All other layers are fully shared.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        ensemble_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        dims = [input_dim] + hidden_dims

        # First layer with MiniEnsemble
        self.first = MiniEnsembleLinear(
            input_dim,
            hidden_dims[0],
            ensemble_size,
            random_init=True,
        )

        # Remaining layers are fully shared
        self.blocks = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.blocks.append(
                SharedBlock(
                    dims[i],
                    dims[i + 1],
                    dropout,
                )
            )

        self.head = nn.Linear(hidden_dims[-1], output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, input_dim)
        x = repeat(x, "b d -> b e d", e=self.first.ensemble_size)

        x = self.first(x)
        x = F.relu(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.head(x)
        return x
