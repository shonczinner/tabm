# Example usage of TabM for regression with y scaling

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tabm import preprocess, train
import numpy as np

# -------------------------------
# Load and split data
# -------------------------------
X, y = fetch_california_housing(return_X_y=True)
X = X.astype(np.float32)
y = y.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, train_size=0.8, random_state=42
)

# -------------------------------
# Preprocess: numeric + PLE + y scaling
# -------------------------------
pre = preprocess.Preprocessor(
    n_bins=100,
    noise=1e-3,
    standardize_y=True,
    random_state=42
)

# Fit and transform training set
X_train_num, _, y_train_std = pre.fit_transform(X_train, y_train)
# Transform validation and test sets
X_val_num, _, y_val_std = pre.transform(X_val, y_val)
X_test_num, _, y_test_std = pre.transform(X_test, y_test)

# -------------------------------
# Initialize and train TabMmini
# -------------------------------
pipe = train.Trainer(
    task='regression',
    n_epochs=150,
    ensemble_size=32,
    is_mini=True,
    batch_size=256,
    use_ple=False
)

pipe.fit(X_train_num, None, y_train_std, 
         X_val_num, None, y_val_std)

# -------------------------------
# Predict and rescale to original y
# -------------------------------
pred_std = pipe.predict(X_test_num, None)
pred = pred_std * pre.y_std + pre.y_mean  # reverse standardization

rmse = np.sqrt(((pred - y_test)**2).mean())
print("Test RMSE:", rmse)

# -------------------------------
# Plot training and validation losses
# -------------------------------
pipe.plot_losses()
