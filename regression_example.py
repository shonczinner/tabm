# Contains example usage of TabM for a regression task with y scaling

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tabm import preprocess, train
import numpy as np

# Load and split data
X, y = fetch_california_housing(return_X_y=True)
X = X.astype(np.float32)
y = y.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=42)

# Preprocess with combined Preprocessor and y scaling
pre = preprocess.Preprocessor(n_bins=100, noise=1e-3, standardize_y=True, random_state=42)
X_train_ple, y_train_std = pre.fit_transform(X_train, y_train)
X_val_ple, y_val_std = pre.transform(X_val, y_val)
X_test_ple, y_test_std = pre.transform(X_test, y_test)

# Train TabM
pipe = train.Trainer(
    task='regression',
    n_epochs=150,
    ensemble_size=32,
    is_mini=True
)

pipe.fit(X_train_ple, y_train_std, X_val_ple, y_val_std)

# Predict and rescale predictions to original y scale
pred_std = pipe.predict(X_test_ple)
# Reverse y standardization
pred = pred_std * pre.y_std + pre.y_mean

rmse = np.sqrt(((pred - y_test)**2).mean())
print("Test RMSE:", rmse)

# Plot losses
pipe.plot_losses()
