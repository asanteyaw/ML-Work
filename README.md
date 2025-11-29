# ML‑Work

This repository contains a collection of **quantitative‑finance and machine‑learning research projects** developed as part of ongoing research and experimentation workflow in **stochastic volatility, forecasting, risk modeling, and numerical methods**. 
This repository contains a collection of machine‑learning and quantitative‑finance projects developed as part of an ongoing research and experimentation workflow.  
Each sub‑project inside **ML‑Work** is self‑contained, well‑structured, and focused on a specific modelling or statistical problem. The goal of the repository is to maintain clear, reproducible, and well‑documented workflows in both Python and C++ (LibTorch) across multiple research streams.

The repository reflects active research in:
- volatility forecasting and regime modeling  
- hybrid ML–quant neural architectures  
- neural volatility integration  
- calibration and estimation (MLE and likelihood‑free)  
- Monte Carlo simulation and derivative pricing  
- sequence models (GRU, LSTM, TCN, Transformers)

---

## 📁 Repository Structure

```
ML-Work/
│
├── 00_term_deposit/
│   ├── data/
│   ├── notebooks/
│   ├── models/
│   └── README.md
│
├── 01_vol_nn_integration/
│   ├── include/
│   ├── src/
│   ├── models/
│   ├── CMakeLists.txt
│   └── README.md
│
└── 02_non_likelihood/
    ├── MLE/
        ├── include/
        ├── src/
        ├── CMakeLists.txt
    ├── TransformerGARCH/
        ├── include/
        ├── src/
        ├── libtorch_tft/
        ├── CMakeLists.txt
    └── README.md
```

Each directory contains its own README describing objectives, methodology, and instructions. The directory structure may change
due to growing commits.

---

## 📌 Project Summaries

### **00_term_deposit**
A classical supervised‑learning workflow used to validate ML tooling and ensure consistent feature‑engineering, model evaluation, and interpretability workflows before applying similar techniques to financial time‑series tasks.

Explores:
- categorical encoding and feature engineering  
- baseline + advanced ML models (LogReg, RF, XGBoost, NN)  
- interpretability methods and validation strategy  

---

### **01_vol_nn_integration**
A research‑oriented project developing neural extensions of classic volatility models, including:
- Heston–Nandi (HN) GARCH
- Component HN (CHN)
- Neural augmentation via GRU/LSTM layers
- Fully differentiable likelihood‑based estimation
- Libtorch based benchmark estimation (HN/CHN)
- Monte Carlo simulation for pricing and forecasting

This project merges econometric models with neural sequence models, enabling richer volatility dynamics and end‑to‑end statistical estimation.

---

### **02_non_likelihood**
A sandbox focused on alternative estimation and inference paradigms outside traditional maximum likelihood.  
This may include:
- Temporal Fusion transformer model
- Novel loss functions
- Simulation‑based or likelihood‑free approaches
- Robust / heavy‑tail models

---

## 🔧 Requirements & Setup

Some projects use Python ≥ 3.10, other are base on C++ and Libtorch (Pytorch C++ API).  
Recommended setup:

```
conda create -n mlwork python=3.10
conda activate mlwork
pip install -r requirements.txt
```

Each subfolder may include its own `requirements.txt` depending on the methods used.

---

## ✨ Goals of the Repository

- Maintain clean, modular, and research‑grade code.
- Allow easy comparison between classical statistical models and modern machine‑learning architectures.
- Provide a reproducible workflow for experiments and thesis‑related development.
- Serve as an evolving archive of all modelling attempts, tests, and exploratory work.

---

## 📜 License
This repository is for academic research and personal experimentation.  
Use and distribution should follow the terms described in each subfolder (if present).

---

## 👨🏾‍💻 Author
Maintained as part of ongoing research in **quantitative finance, stochastic modeling, and applied machine learning**.


