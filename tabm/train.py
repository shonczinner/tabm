import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from tabm import tabm
from tabm import losses  # regression_loss and classification_loss


class Trainer:
    """
    Flexible Trainer for TabM ensemble models (TabM / TabMmini) with optional PLE.
    Tracks per-member training loss, ensemble-averaged training loss,
    and averaged validation/inference loss.
    """
    def __init__(self, task='regression', hidden_dims=[512, 512, 512],
                 d_embedding=16, lr=1e-3, batch_size=32, n_epochs=150,
                 ensemble_size=32, dropout=0.1, device=None, is_mini=False,
                 use_ple=True):
        self.task = task
        self.hidden_dims = hidden_dims
        self.d_embedding = d_embedding
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ensemble_size = ensemble_size
        self.dropout = dropout
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_mini = is_mini
        self.use_ple = use_ple

        self.input_dim = None
        self.n_bins = 0
        self.output_dim = None
        self.model = None
        self.optimizer = None

        self.train_losses = []
        self.train_losses_ensemble_avg = []
        self.val_losses = []

    def fit(self, X_num, X_ple=None, y_train=None, X_val_num=None, X_val_ple=None, y_val=None):
        X_num = torch.tensor(X_num, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        self.input_dim = X_num.shape[1]

        # Determine n_bins if PLE is used
        if self.use_ple:
            if X_ple is None:
                raise ValueError("X_ple must be provided when use_ple=True")
            X_ple = torch.tensor(X_ple, dtype=torch.float32).to(self.device)
            self.n_bins = X_ple.shape[2]
            dataset = TensorDataset(X_num, X_ple, y_train)
        else:
            dataset = TensorDataset(X_num, y_train)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Output dimension
        if self.task == 'regression':
            self.output_dim = 1 if y_train.ndim == 1 else y_train.shape[1]
        else:
            self.output_dim = int(y_train.max().item()) + 1

        # Initialize model if not already
        if self.model is None:
            ModelClass = tabm.TabMmini if self.is_mini else tabm.TabM
            self.model = ModelClass(
                n_features=self.input_dim,
                n_bins=self.n_bins if self.use_ple else 0,
                d_embedding=self.d_embedding,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                ensemble_size=self.ensemble_size,
                dropout=self.dropout,
                use_ple=self.use_ple
            ).to(self.device)

        loss_fn = losses.regression_loss if self.task == 'regression' else losses.classification_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.n_epochs):
            self.model.train()
            batch_losses = []

            for batch in loader:
                self.optimizer.zero_grad()
                if self.use_ple:
                    xb_num, xb_ple, yb = batch
                    pred = self.model(xb_num, xb_ple)
                else:
                    xb_num, yb = batch
                    pred = self.model(xb_num)

                # Per-member loss
                loss1 = loss_fn(pred, yb, training=True)
                loss1_mean = loss1.mean()
                loss1_mean.backward()
                self.optimizer.step()

                with torch.no_grad():
                    # Ensemble-average loss
                    loss2 = loss_fn(pred, yb, training=False)
                    batch_losses.append((loss1_mean.item(), loss2.mean().item()))

            batch_losses = np.array(batch_losses)
            self.train_losses.append(batch_losses[:, 0].mean())
            self.train_losses_ensemble_avg.append(batch_losses[:, 1].mean())

            # Validation
            if X_val_num is not None and y_val is not None:
                val_loss = self.evaluate(X_val_num, X_val_ple, y_val, loss_fn)
                self.val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{self.n_epochs}, "
                      f"Train Loss: {self.train_losses[-1]:.4f}, "
                      f"Train Loss (ensemble avg): {self.train_losses_ensemble_avg[-1]:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.n_epochs}, "
                      f"Train Loss: {self.train_losses[-1]:.4f}, "
                      f"Train Loss (ensemble avg): {self.train_losses_ensemble_avg[-1]:.4f}")

    def predict(self, X_num, X_ple=None):
        self.model.eval()
        X_num_tensor = torch.tensor(X_num, dtype=torch.float32).to(self.device)

        if self.use_ple:
            if X_ple is None:
                raise ValueError("X_ple must be provided when model was trained with PLE")
            X_ple_tensor = torch.tensor(X_ple, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = self.model(X_num_tensor, X_ple_tensor) if self.use_ple else self.model(X_num_tensor)
            if self.task == 'regression':
                return pred.mean(dim=1).cpu().numpy().squeeze()
            else:
                probs = torch.softmax(pred, dim=-1)
                return probs.mean(dim=1).cpu().numpy()

    def evaluate(self, X_num, X_ple=None, y=None, loss_fn=None):
        self.model.eval()
        X_num_tensor = torch.tensor(X_num, dtype=torch.float32).to(self.device)

        if self.use_ple:
            if X_ple is None:
                raise ValueError("X_ple must be provided when model was trained with PLE")
            X_ple_tensor = torch.tensor(X_ple, dtype=torch.float32).to(self.device)

        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        loss_fn = loss_fn or (losses.regression_loss if self.task=='regression' else losses.classification_loss)

        with torch.no_grad():
            pred = self.model(X_num_tensor, X_ple_tensor) if self.use_ple else self.model(X_num_tensor)
            return loss_fn(pred, y_tensor, training=False).mean().item()

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
