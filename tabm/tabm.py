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
    Returns: (batch, ensemble, out_features)
    """
    def __init__(self, in_features, out_features, ensemble_size, bias=True, random_init=False):
        super().__init__()
        self.ensemble_size = ensemble_size

        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.R = nn.Parameter(torch.ones(ensemble_size, in_features))
        self.S = nn.Parameter(torch.ones(ensemble_size, out_features))
        if bias:
            self.B = nn.Parameter(torch.zeros(ensemble_size, out_features))
        else:
            self.register_parameter("B", None)

        if random_init:
            self.R.data = torch.randint(0, 2, self.R.shape, dtype=torch.float32) * 2 - 1
            self.S.data = torch.randint(0, 2, self.S.shape, dtype=torch.float32) * 2 - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, ensemble, in_features)
        returns: (batch, ensemble, out_features)
        """
        x = x * rearrange(self.R, "e d -> 1 e d")
        x = rearrange(x, "b e d -> (b e) d")
        x = self.weight(x)
        x = rearrange(x, "(b e) o -> b e o", e=self.ensemble_size)
        x = x * rearrange(self.S, "e o -> 1 e o")
        if self.B is not None:
            x = x + rearrange(self.B, "e o -> 1 e o")
        return x

# =========================
# MiniEnsemble Linear
# =========================
class MiniEnsembleLinear(nn.Module):
    """
    MiniEnsemble linear layer:
    l_ME(X) = (X ⊙ R) W

    X: (batch, ensemble, in_features)
    Returns: (batch, ensemble, out_features)
    """
    def __init__(self, in_features, out_features, ensemble_size, random_init=True):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.R = nn.Parameter(torch.ones(ensemble_size, in_features))
        if random_init:
            self.R.data = torch.randint(0, 2, self.R.shape, dtype=torch.float32) * 2 - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, ensemble, in_features)
        returns: (batch, ensemble, out_features)
        """
        x = x * rearrange(self.R, "e d -> 1 e d")
        x = rearrange(x, "b e d -> (b e) d")
        x = self.weight(x)
        x = rearrange(x, "(b e) o -> b e o", e=self.ensemble_size)
        return x

# =========================
# Blocks
# =========================
class TabMBlock(nn.Module):
    """BatchEnsemble block: Linear → ReLU → Dropout"""
    def __init__(self, in_features, out_features, ensemble_size, dropout, random_init=False):
        super().__init__()
        self.linear = BatchEnsembleLinear(in_features, out_features, ensemble_size, random_init=random_init)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, ensemble, in_features)
        returns: (batch, ensemble, out_features)
        """
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class SharedBlock(nn.Module):
    """Fully shared block (no adapters): Linear → ReLU → Dropout"""
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, ensemble, in_features)
        returns: (batch, ensemble, out_features)
        """
        batch, ensemble, _ = x.shape
        x = rearrange(x, "b e d -> (b e) d")
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = rearrange(x, "(b e) d -> b e d", b=batch, e=ensemble)
        return x

# =========================
# Piecewise Linear Embeddings
# =========================
class PiecewiseLinearEmbeddings(nn.Module):
    """
    Per-feature piecewise-linear embeddings.
    Input numeric features + PLE:
    x_num: (batch, n_features)
    x_ple: (batch, n_features, n_bins)
    Returns: (batch, n_features, d_embedding)
    """
    def __init__(self, n_features, n_bins, d_embedding, activation=True):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding

        self.num_linears = nn.ModuleList([nn.Linear(1, d_embedding) for _ in range(n_features)])
        self.ple_linears = nn.ModuleList([nn.Linear(n_bins, d_embedding, bias=False) for _ in range(n_features)])
        self.activation = nn.ReLU() if activation else None

    def forward(self, x_num: torch.Tensor, x_ple: torch.Tensor) -> torch.Tensor:
        """
        x_num: (batch, n_features)
        x_ple: (batch, n_features, n_bins)
        returns: (batch, n_features * d_embedding)
        """
        embeddings = []
        for i in range(self.n_features):
            ple_emb = self.ple_linears[i](x_ple[:, i, :])
            num_emb = self.num_linears[i](x_num[:, i:i+1])
            embeddings.append(ple_emb + num_emb)
        out = torch.cat(embeddings, dim=-1)
        if self.activation is not None:
            out = self.activation(out)
        return out
    
class CategoricalEmbeddings(nn.Module):
    """
    x_cat: (batch, n_cat)
    Returns: (batch, n_cat * d_cat)
    """
    def __init__(self, cardinalities, d_cat):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, d_cat) for card in cardinalities]
        )

    def forward(self, x_cat):
        outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(outs, dim=-1)


# =========================
# TabM
# =========================
class TabM(nn.Module):
    def __init__(
        self,
        n_num,
        n_bins,
        n_cat,
        cat_cardinalities,
        d_embedding,
        d_cat,
        hidden_dims,
        output_dim,
        ensemble_size,
        dropout=0.0,
        use_ple=True,
    ):
        super().__init__()

        self.n_num = n_num
        self.n_cat = n_cat
        self.ensemble_size = ensemble_size

        self.use_ple = use_ple and n_num > 0

        input_dim = 0

        # Numerical features
        if n_num > 0:
            if self.use_ple:
                self.ple = PiecewiseLinearEmbeddings(n_num, n_bins, d_embedding)
                input_dim += n_num * d_embedding + n_num
            else:
                self.ple = None
                input_dim += n_num
        else:
            self.ple = None

        # Categorical features
        if n_cat > 0:
            self.cat = CategoricalEmbeddings(cat_cardinalities, d_cat)
            input_dim += n_cat * d_cat
        else:
            self.cat = None

        dims = [input_dim] + hidden_dims
        self.blocks = nn.ModuleList(
            TabMBlock(
                dims[i],
                dims[i + 1],
                ensemble_size,
                dropout,
                random_init=(i == 0),
            )
            for i in range(len(hidden_dims))
        )

        self.head = BatchEnsembleLinear(hidden_dims[-1], output_dim, ensemble_size)

    def forward(self, x_num=None, x_ple=None, x_cat=None):
        feats = []

        if self.n_num > 0:
            assert x_num is not None, "x_num is required when n_num > 0"
            if self.use_ple:
                assert x_ple is not None, "x_ple is required when use_ple=True"
                feats.append(x_num)
                feats.append(self.ple(x_num, x_ple))
            else:
                feats.append(x_num)

        if self.n_cat > 0:
            assert x_cat is not None, "x_cat is required when n_cat > 0"
            feats.append(self.cat(x_cat))

        x = torch.cat(feats, dim=-1)
        x = repeat(x, "b d -> b e d", e=self.ensemble_size)

        for block in self.blocks:
            x = block(x)

        return self.head(x)


# =========================
# TabMmini
# =========================
class TabMMini(nn.Module):
    def __init__(
        self,
        n_num,
        n_bins,
        n_cat,
        cat_cardinalities,
        d_embedding,
        d_cat,
        hidden_dims,
        output_dim,
        ensemble_size,
        dropout=0.0,
        use_ple=True,
    ):
        super().__init__()

        self.n_num = n_num
        self.n_cat = n_cat
        self.ensemble_size = ensemble_size

        self.use_ple = use_ple and n_num > 0

        input_dim = 0

        if n_num > 0:
            if self.use_ple:
                self.ple = PiecewiseLinearEmbeddings(n_num, n_bins, d_embedding)
                input_dim += n_num * d_embedding + n_num
            else:
                self.ple = None
                input_dim += n_num
        else:
            self.ple = None

        if n_cat > 0:
            self.cat = CategoricalEmbeddings(cat_cardinalities, d_cat)
            input_dim += n_cat * d_cat
        else:
            self.cat = None

        dims = [input_dim] + hidden_dims
        self.blocks = nn.ModuleList(
            nn.Sequential(
                MiniEnsembleLinear(dims[i], dims[i + 1], ensemble_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for i in range(len(hidden_dims))
        )

        self.head = MiniEnsembleLinear(hidden_dims[-1], output_dim, ensemble_size)

    def forward(self, x_num=None, x_ple=None, x_cat=None):
        feats = []

        if self.n_num > 0:
            assert x_num is not None
            if self.use_ple:
                assert x_ple is not None
                feats.append(x_num)
                feats.append(self.ple(x_num, x_ple))
            else:
                feats.append(x_num)

        if self.n_cat > 0:
            assert x_cat is not None
            feats.append(self.cat(x_cat))

        x = torch.cat(feats, dim=-1)
        x = repeat(x, "b d -> b e d", e=self.ensemble_size)

        for block in self.blocks:
            x = block(x)

        return self.head(x)
