# Example usage of TabMmini with mixed numeric + categorical features

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tabm import preprocess, train
import numpy as np
import torch

# -------------------------------
# Load Adult Census Income dataset
# -------------------------------
# fetch_openml automatically returns pandas DataFrame
adult = fetch_openml('adult', version=2, as_frame=True)
X = adult.data
y = adult.target

# Encode target as integer (0: <=50K, 1: >50K)
y = LabelEncoder().fit_transform(y)
y = y.astype(np.int64)

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# -------------------------------
# Simple preprocessing
# -------------------------------
# Encode categorical columns as integer indices
cat_cardinalities = []
X_cat_encoded = np.zeros((X.shape[0], len(categorical_cols)), dtype=np.int64)
for i, col in enumerate(categorical_cols):
    le = LabelEncoder()
    X_cat_encoded[:, i] = le.fit_transform(X[col])
    cat_cardinalities.append(X[col].nunique())

# Numeric features as float32
X_num = X[numeric_cols].astype(np.float32).to_numpy()

# Split train/val/test
X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
    X_num, X_cat_encoded, y, train_size=0.8, random_state=42
)
X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
    X_num_train, X_cat_train, y_train, train_size=0.8, random_state=42
)

# -------------------------------
# Preprocess numeric features with optional PLE
# -------------------------------
pre = preprocess.Preprocessor(n_bins=10, noise=1e-3, random_state=42)
X_num_train_std, X_train_ple = pre.fit_transform(X_num_train)
X_num_val_std, X_val_ple = pre.transform(X_num_val)
X_num_test_std, X_test_ple = pre.transform(X_num_test)

# -------------------------------
# Initialize and train TabMMini
# -------------------------------
pipe = train.Trainer(
    task='classification',
    n_epochs=50,
    ensemble_size=32,
    is_mini=True,
    batch_size=256,
    dropout=0.3
)

pipe.fit(
    X_num=X_num_train_std,
    X_ple=X_train_ple,
    X_cat=X_cat_train,
    y_train=y_train,
    X_val_num=X_num_val_std,
    X_val_ple=X_val_ple,
    X_val_cat=X_cat_val,
    y_val=y_val
)

# -------------------------------
# Predict probabilities and classes
# -------------------------------
probs = pipe.predict(X_num=X_num_test_std,
                     X_ple=X_test_ple,
                     X_cat=X_cat_test)
pred_classes = np.argmax(probs, axis=1)

# Compute accuracy
accuracy = (pred_classes == y_test).mean()
print("Test Accuracy:", accuracy)

# -------------------------------
# Plot training and validation losses
# -------------------------------
pipe.plot_losses()
