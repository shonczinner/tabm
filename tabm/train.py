import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from tabm import tabm
from tabm import losses  # regression_loss and classification_loss


class Trainer:
    """
    Flexible Trainer for TabM / TabMMini ensemble models with optional PLE.
    Tracks per-member training loss, ensemble-averaged training loss,
    and averaged validation/inference loss.
    Handles numeric features, categorical features, PLE, or any combination.
    """
    def __init__(self, task='regression', hidden_dims=[512, 512, 512],
                 d_embedding=16,d_cat=8, lr=1e-3, batch_size=32, n_epochs=150,
                 ensemble_size=32, dropout=0.1, device=None, is_mini=False,
                 use_ple=True):
        self.task = task
        self.hidden_dims = hidden_dims
        self.d_embedding = d_embedding
        self.d_cat = d_cat
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ensemble_size = ensemble_size
        self.dropout = dropout
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_mini = is_mini
        self.use_ple = use_ple

        self.n_num = 0
        self.n_cat = 0
        self.n_bins = 0
        self.output_dim = None
        self.cat_cardinalities = None
        self.model = None
        self.optimizer = None

        self.train_losses = []
        self.train_losses_ensemble_avg = []
        self.val_losses = []

    def fit(self, X_num=None, X_ple=None, X_cat=None, y_train=None,
            X_val_num=None, X_val_ple=None, X_val_cat=None, y_val=None):

        # -------------------------
        # Prepare tensors
        # -------------------------
        tensors = []

        # Numeric features
        if X_num is not None:
            self.n_num = X_num.shape[1]
            X_num_tensor = torch.tensor(X_num, dtype=torch.float32).to(self.device)
            tensors.append(X_num_tensor)
        else:
            self.n_num = 0
            X_num_tensor = None

        # PLE
        if self.use_ple and X_ple is not None:
            X_ple_tensor = torch.tensor(X_ple, dtype=torch.float32).to(self.device)
            self.n_bins = X_ple_tensor.shape[2]
            tensors.append(X_ple_tensor)
        else:
            X_ple_tensor = None
            self.n_bins = 0

        # Categorical features
        if X_cat is not None:
            self.n_cat = X_cat.shape[1]
            X_cat_tensor = torch.tensor(X_cat, dtype=torch.long).to(self.device)
            self.cat_cardinalities = [int(X_cat_tensor[:, i].max().item()) + 1
                                      for i in range(self.n_cat)]
            tensors.append(X_cat_tensor)
        else:
            self.n_cat = 0
            X_cat_tensor = None
            self.cat_cardinalities = None

        # Labels
        if self.task == 'regression':
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        else:
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        tensors.append(y_train_tensor)

        # -------------------------
        # Dataset & loader
        # -------------------------
        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Output dimension
        if self.task == 'regression':
            self.output_dim = 1 if y_train_tensor.ndim == 1 else y_train_tensor.shape[1]
        else:
            self.output_dim = int(y_train_tensor.max().item()) + 1

        # -------------------------
        # Initialize model
        # -------------------------
        if self.model is None:
            ModelClass = tabm.TabMMini if self.is_mini else tabm.TabM
            self.model = ModelClass(
                n_num=self.n_num,
                n_bins=self.n_bins,
                n_cat=self.n_cat,
                cat_cardinalities=self.cat_cardinalities or [],
                d_embedding=self.d_embedding,
                d_cat=self.d_cat,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                ensemble_size=self.ensemble_size,
                dropout=self.dropout,
                use_ple=self.use_ple
            ).to(self.device)

        loss_fn = losses.regression_loss if self.task == 'regression' else losses.classification_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # -------------------------
        # Training loop
        # -------------------------
        for epoch in range(self.n_epochs):
            self.model.train()
            batch_losses = []

            for batch in loader:
                self.optimizer.zero_grad()

                # Unpack batch dynamically
                idx = 0
                xb_num = xb_ple = xb_cat = None

                if self.n_num > 0:
                    xb_num = batch[idx]
                    idx += 1
                if self.use_ple and self.n_bins > 0:
                    xb_ple = batch[idx]
                    idx += 1
                if self.n_cat > 0:
                    xb_cat = batch[idx]
                    idx += 1
                yb = batch[idx]

                # Forward pass
                pred = self.model(xb_num, xb_ple, xb_cat)

                # Per-member loss
                loss1 = loss_fn(pred, yb, training=True)
                loss1_mean = loss1.mean()
                loss1_mean.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                with torch.no_grad():
                    loss2 = loss_fn(pred, yb, training=False)
                    batch_losses.append((loss1_mean.item(), loss2.mean().item()))

            batch_losses = np.array(batch_losses)
            self.train_losses.append(batch_losses[:, 0].mean())
            self.train_losses_ensemble_avg.append(batch_losses[:, 1].mean())

            # Validation
            if y_val is not None:
                val_loss = self.evaluate(X_val_num, X_val_ple, X_val_cat, y_val, loss_fn)
                self.val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{self.n_epochs}, "
                      f"Train Loss: {self.train_losses[-1]:.4f}, "
                      f"Train Loss (ensemble avg): {self.train_losses_ensemble_avg[-1]:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.n_epochs}, "
                      f"Train Loss: {self.train_losses[-1]:.4f}, "
                      f"Train Loss (ensemble avg): {self.train_losses_ensemble_avg[-1]:.4f}")

    # -------------------------
    # Prediction
    # -------------------------
    def predict(self, X_num=None, X_ple=None, X_cat=None):
        self.model.eval()

        xb_num = torch.tensor(X_num, dtype=torch.float32).to(self.device) if X_num is not None else None
        xb_ple = torch.tensor(X_ple, dtype=torch.float32).to(self.device) if X_ple is not None else None
        xb_cat = torch.tensor(X_cat, dtype=torch.long).to(self.device) if X_cat is not None else None

        with torch.no_grad():
            pred = self.model(xb_num, xb_ple, xb_cat)
            if self.task == 'regression':
                return pred.mean(dim=1).cpu().numpy().squeeze()
            else:
                probs = torch.softmax(pred, dim=-1)
                return probs.mean(dim=1).cpu().numpy()

    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate(self, X_num=None, X_ple=None, X_cat=None, y=None, loss_fn=None):
        self.model.eval()

        xb_num = torch.tensor(X_num, dtype=torch.float32).to(self.device) if X_num is not None else None
        xb_ple = torch.tensor(X_ple, dtype=torch.float32).to(self.device) if X_ple is not None else None
        xb_cat = torch.tensor(X_cat, dtype=torch.long).to(self.device) if X_cat is not None else None

        if self.task == 'regression':
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        loss_fn = loss_fn or (losses.regression_loss if self.task=='regression' else losses.classification_loss)

        with torch.no_grad():
            pred = self.model(xb_num, xb_ple, xb_cat)
            return loss_fn(pred, y_tensor, training=False).mean().item()

    # -------------------------
    # Plot losses
    # -------------------------
    def plot_losses(self):
        plt.figure(figsize=(6,4))
        plt.plot(self.train_losses, label='Train Loss (per-member)')
        plt.plot(self.train_losses_ensemble_avg, label='Train Loss (ensemble avg)')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training / Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
