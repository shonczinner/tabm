import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from tabm import tabm
from tabm import losses  # updated regression_loss and classification_loss


class Trainer:
    """
    Trainer for TabM ensemble models.
    Tracks per-member training loss, ensemble-averaged training loss,
    and averaged validation/inference loss.
    """
    def __init__(self, task='regression', hidden_dims=[512, 512, 512],
                 lr=1e-3, batch_size=32, n_epochs=150, ensemble_size=32,
                 device=None, is_mini=False):
        self.task = task
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ensemble_size = ensemble_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_mini = is_mini

        self.input_dim = None
        self.output_dim = None
        self.model = None
        self.optimizer = None

        self.train_losses = []                # per-member averaged
        self.train_losses_ensemble_avg = []   # ensemble-avg output then loss
        self.val_losses = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = torch.tensor(X_train).to(self.device)
        y_train = torch.tensor(y_train).to(self.device)

        self.input_dim = X_train.shape[1]
        if self.task == 'regression':
            self.output_dim = 1 if y_train.ndim == 1 else y_train.shape[1]
        else:
            self.output_dim = int(y_train.max().item()) + 1

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.model is None:
            if not self.is_mini:
                self.model = tabm.TabM(
                    input_dim=self.input_dim,
                    hidden_dims=self.hidden_dims,
                    output_dim=self.output_dim,
                    ensemble_size=self.ensemble_size
                ).to(self.device)
            else:
                self.model = tabm.TabMmini(
                    input_dim=self.input_dim,
                    hidden_dims=self.hidden_dims,
                    output_dim=self.output_dim,
                    ensemble_size=self.ensemble_size
                ).to(self.device)


        loss_fn = losses.regression_loss if self.task=='regression' else losses.classification_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.n_epochs):
            self.model.train()
            batch_losses = []

            for xb, yb in loader:
                self.optimizer.zero_grad()
                pred = self.model(xb)

                # Loss 1: per-member
                loss1 = loss_fn(pred, yb, training=True)
                loss1_mean = loss1.mean()  # average over batch & ensemble
                loss1_mean.backward()
                self.optimizer.step()

                # Loss 2: ensemble-avg
                loss2 = loss_fn(pred, yb, training=False)
                batch_losses.append((loss1_mean.item(), loss2.mean().item()))

            batch_losses = np.array(batch_losses)
            self.train_losses.append(batch_losses[:,0].mean())
            self.train_losses_ensemble_avg.append(batch_losses[:,1].mean())

            # Validation loss
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val, loss_fn)
                self.val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{self.n_epochs}, "
                      f"Train Loss: {self.train_losses[-1]:.4f}, "
                      f"Train Loss (ensemble avg): {self.train_losses_ensemble_avg[-1]:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.n_epochs}, "
                      f"Train Loss: {self.train_losses[-1]:.4f}, "
                      f"Train Loss (ensemble avg): {self.train_losses_ensemble_avg[-1]:.4f}")

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X).to(self.device)
        with torch.no_grad():
            pred = self.model(X_tensor)
            if self.task=='regression':
                return pred.mean(dim=1).cpu().numpy().squeeze()
            else:
                probs = torch.softmax(pred, dim=-1)
                return probs.mean(dim=1).cpu().numpy()

    def evaluate(self, X, y, loss_fn=None):
        self.model.eval()
        X_tensor = torch.tensor(X).to(self.device)
        y_tensor = torch.tensor(y).to(self.device)
        loss_fn = loss_fn or (losses.regression_loss if self.task=='regression' else losses.classification_loss)
        with torch.no_grad():
            pred = self.model(X_tensor)
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

