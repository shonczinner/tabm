# Example usage of TabMmini for a classification task

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tabm import preprocess, train
import numpy as np

# -------------------------------
# Load dataset
# -------------------------------
X, y = load_breast_cancer(return_X_y=True)
X = X.astype(np.float32)
y = y.astype(np.int64)  # integer class labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, train_size=0.8, random_state=42
)

# -------------------------------
# Preprocess features only (numeric + PLE)
# -------------------------------
pre = preprocess.Preprocessor(
    n_bins=4,
    n_quantiles=20,
    noise=1e-3,
    random_state=42
)
# Fit-transform returns numeric features and PLE features
X_train_num, X_train_ple = pre.fit_transform(X_train)
X_val_num, X_val_ple = pre.transform(X_val)
X_test_num, X_test_ple = pre.transform(X_test)

# -------------------------------
# Initialize and train TabMmini for classification
# -------------------------------
pipe = train.Trainer(
    task='classification',
    n_epochs=100,
    ensemble_size=8,
    is_mini=True,
    batch_size=32,
    d_embedding=5,
    dropout=0.2
)

pipe.fit(
    X_num=X_train_num,
    X_ple=X_train_ple,
    y_train=y_train,
    X_val_num=X_val_num,
    X_val_ple=X_val_ple,
    y_val=y_val
)

# -------------------------------
# Predict probabilities and classes
# -------------------------------
probs = pipe.predict(X_num=X_test_num, X_ple=X_test_ple)  # averaged ensemble probabilities
pred_classes = np.argmax(probs, axis=1)

# Compute accuracy
accuracy = (pred_classes == y_test).mean()
print("Test Accuracy:", accuracy)

# -------------------------------
# Plot training and validation losses
# -------------------------------
pipe.plot_losses()
