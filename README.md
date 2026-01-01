# TabM (Unofficial Implementation)

This repository contains an **unofficial Python implementation** of **TabM**, a parameter-efficient ensembling model for tabular deep learning.

Based on the paper:

**TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling**  
Yury Gorishniy, Akim Kotelnikov, Artem Babenko  
arXiv:2410.24210 (v3, Feb 18 2025)  
https://doi.org/10.48550/arXiv.2410.24210

Official implementation: [yandex-research/tabm](https://github.com/yandex-research/tabm)

---

## Project Status

**Work in progress.** Current limitations:

- Categorical features are **not implemented**  
- Focus on core TabM behavior, not full benchmark parity  

---

## Structure

- `tabm/` — core model  
- `tests/` — unit tests  
- `classification_example.py` — classification example  
- `regression_example.py` — regression example  

---

## Running Tests

```
python -m pytest
```


## Running Examples

We provide some examples for using TabM: 

### Regression Example (California Housing)

```
python examples/regression_example.py
```

Expected output:
```
Epoch 1/150, Train Loss: 0.5501, Train Loss (ensemble avg): 0.5183, Val Loss: 0.3080
...
Epoch 150/150, Train Loss: 0.0743, Train Loss (ensemble avg): 0.0466, Val Loss: 0.1548
Test RMSE: 0.4307975
```

This trains a TabMmini model on the California Housing dataset, standardizes the targets, and reports test RMSE.

### Regression (without piecewise linear embeddings) Example (California Housing)

```
python examples/regression_example_no_ple.py
```

Expected output:
```
Epoch 1/150, Train Loss: 0.5095, Train Loss (ensemble avg): 0.4798, Val Loss: 0.3119
...
Epoch 150/150, Train Loss: 0.1135, Train Loss (ensemble avg): 0.0863, Val Loss: 0.1807
Test RMSE: 0.4721563
```

This trains a TabMmini model on the California Housing dataset, standardizes the targets, and reports test RMSE.

### Classification Example (Breast Cancer)
python examples/classification_example.py


Expected output:
```
Epoch 1/100, Train Loss: 0.6377, Train Loss (ensemble avg): 0.6552, Val Loss: 0.5696
...
Epoch 100/100, Train Loss: 0.0010, Train Loss (ensemble avg): 0.3138, Val Loss: 0.3479
Test Accuracy: 0.9824561403508771
```

This trains a TabMmini model for classification using PLE features and reports ensemble-averaged probabilities.

### Notes

- Losses are plotted automatically after training with pipe.plot_losses().

- Regression example automatically rescales predictions to the original target scale.

- Classification example averages ensemble outputs before computing predicted classes.


## Install Dependencies

```
pip install -r requirements.txt
```

Note: PyTorch should be installed according to your system and CUDA version. 
See PyTorch Get Started for the recommended command.

## Disclaimer

This is an unofficial reimplementation for learning and experimentation.
It is not affiliated with the original authors.