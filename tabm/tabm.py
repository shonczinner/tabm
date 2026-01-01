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


# ----------------------------
# Piecewise Linear Embeddings
# ----------------------------
class PiecewiseLinearEmbeddings(nn.Module):
    """
    Version B piecewise-linear embeddings for TabM.

    Each numeric feature now has its own linear projection to avoid shape mismatch.
    """
    def __init__(self, n_features: int, n_bins: int, d_embedding: int, activation: bool = True):
        super().__init__()
        self.d_embedding = d_embedding
        self.n_features = n_features

        # Per-feature linear projections for numeric features
        self.linear0 = nn.ModuleList([nn.Linear(1, d_embedding) for _ in range(n_features)])

        # Linear projection of PLE encodings: W·PLE(x)
        self.ple_linear = nn.ModuleList([
            nn.Linear(n_bins, d_embedding, bias=False) for _ in range(n_features)
        ])

        self.activation = nn.ReLU() if activation else None

    def forward(self, x_num: torch.Tensor, x_ple: torch.Tensor):
        batch_size, n_features = x_num.shape
        embeddings = []

        for i in range(n_features):
            ple_emb = self.ple_linear[i](x_ple[:, i, :])      # (batch, d_embedding)
            num_emb = self.linear0[i](x_num[:, i].unsqueeze(1))  # (batch, d_embedding)
            embeddings.append(ple_emb + num_emb)

        out = torch.stack(embeddings, dim=1)  # (batch, n_features, d_embedding)
        if self.activation is not None:
            out = self.activation(out)
        return out


# ----------------------------
# TabM (with optional PLE)
# ----------------------------
class TabM(nn.Module):
    """
    TabM = Piecewise Linear Embeddings + MLP + BatchEnsemble
    PLE is optional. If trained with PLE, it must be provided at inference.
    """
    def __init__(
        self,
        n_features: int,
        n_bins: int | None,
        d_embedding: int,
        hidden_dims: list[int],
        output_dim: int,
        ensemble_size: int,
        dropout: float = 0.0,
        use_ple: bool = True,
    ):
        super().__init__()
        self.use_ple = use_ple
        self.requires_ple = use_ple and n_bins is not None
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.n_bins = n_bins if self.requires_ple else 1

        self.embedding = PiecewiseLinearEmbeddings(
            n_features, self.n_bins, d_embedding
        )

        input_dim = n_features * d_embedding
        dims = [input_dim] + hidden_dims
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.blocks.append(
                TabMBlock(
                    dims[i],
                    dims[i + 1],
                    ensemble_size,
                    dropout,
                    random_init=(i == 0)
                )
            )

        self.head = BatchEnsembleLinear(hidden_dims[-1], output_dim, ensemble_size, random_init=False)

    def forward(self, x_num: torch.Tensor, x_ple: torch.Tensor | None = None):
        if self.requires_ple and x_ple is None:
            raise ValueError("This model was trained with PLE; x_ple cannot be None.")
        if not self.requires_ple and x_ple is None:
            # Use dummy tensor for numeric-only
            x_ple = torch.zeros(x_num.shape[0], x_num.shape[1], 1, device=x_num.device)

        x_emb = self.embedding(x_num, x_ple)
        x_flat = x_emb.flatten(start_dim=1)
        x_flat = repeat(x_flat, "b d -> b e d", e=self.head.ensemble_size)

        for block in self.blocks:
            x_flat = block(x_flat)

        return self.head(x_flat)


# ----------------------------
# TabMmini (with optional PLE)
# ----------------------------
class TabMmini(nn.Module):
    """
    TabMmini = Piecewise Linear Embeddings + MLP + MiniEnsemble
    PLE is optional. If trained with PLE, it must be provided at inference.
    """
    def __init__(
        self,
        n_features: int,
        n_bins: int | None,
        d_embedding: int,
        hidden_dims: list[int],
        output_dim: int,
        ensemble_size: int,
        dropout: float = 0.0,
        use_ple: bool = True,
    ):
        super().__init__()
        self.use_ple = use_ple
        self.requires_ple = use_ple and n_bins is not None
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.n_bins = n_bins if self.requires_ple else 1

        self.embedding = PiecewiseLinearEmbeddings(
            n_features, self.n_bins, d_embedding
        )

        input_dim = n_features * d_embedding
        dims = [input_dim] + hidden_dims

        # First layer with MiniEnsemble
        self.first = MiniEnsembleLinear(input_dim, hidden_dims[0], ensemble_size, random_init=True)

        # Remaining layers are fully shared
        self.blocks = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.blocks.append(SharedBlock(dims[i], dims[i + 1], dropout))

        self.head = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_num: torch.Tensor, x_ple: torch.Tensor | None = None):
        if self.requires_ple and x_ple is None:
            raise ValueError("This model was trained with PLE; x_ple cannot be None.")
        if not self.requires_ple and x_ple is None:
            x_ple = torch.zeros(x_num.shape[0], x_num.shape[1], 1, device=x_num.device)

        x_emb = self.embedding(x_num, x_ple)
        x_flat = x_emb.flatten(start_dim=1)
        x_flat = repeat(x_flat, "b d -> b e d", e=self.first.ensemble_size)

        x_flat = self.first(x_flat)
        x_flat = F.relu(x_flat)
        x_flat = self.dropout(x_flat)

        for block in self.blocks:
            x_flat = block(x_flat)

        x_flat = self.head(x_flat)
        return x_flat
