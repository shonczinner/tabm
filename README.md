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

- No piecewise linear *embeddings* (only encoding fed to MLP)  
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

```bash
python -m pytest
```


## Install Dependencies

```bash
pip install -r requirements.txt
```

Note: PyTorch should be installed according to your system and CUDA version. 
See PyTorch Get Started for the recommended command.

## Disclaimer

This is an unofficial reimplementation for learning and experimentation.
It is not affiliated with the original authors.