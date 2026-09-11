# 🔬 Perovskite Solar Cell Efficiency Predictor

A machine learning model that predicts the **power conversion efficiency (PCE)** of perovskite solar cells directly from condensed matter defect properties. Deployed as an interactive web app on **Hugging Face Spaces**.

## 📌 Overview

Perovskite solar cells (PSCs) have emerged as one of the most promising next-generation photovoltaic technologies, with lab efficiencies exceeding 26%. However, their performance is highly sensitive to **defect chemistry** — subtle changes in band gap, trap states, and grain morphology can shift efficiency by several percentage points.

This project bridges **condensed matter physics** and **machine learning** by training a regression model that maps fundamental defect properties directly to device efficiency, enabling rapid screening of candidate perovskite compositions without expensive DFT or fabrication cycles.

---

## 🎯 Problem Statement

Traditional approaches to predicting perovskite efficiency rely on:
- **First-principles DFT calculations** — accurate but computationally expensive
- **Trial-and-error fabrication** — slow and resource-intensive

**Goal:** Build a fast, data-driven surrogate model that predicts efficiency from four easily-obtainable defect descriptors.

---

## 📊 Input Features

The model takes **four condensed matter defect properties** as input:

| Feature | Symbol | Unit | Physical Meaning |
|---|---|---|---|
| **Band Gap** | E_g | eV | Energy separation between valence and conduction bands |
| **Defect Density** | N_d | cm⁻³ | Concentration of trap states within the band gap |
| **Trap Depth** | E_t | eV | Energy level of traps relative to band edges |
| **Grain Size** | G | nm | Average crystalline grain dimension |

These features capture the dominant recombination pathways (Shockley-Read-Hall), charge transport limitations, and morphological quality of the perovskite absorber layer.

---

## 🤖 Model

| Aspect | Details |
|---|---|
| **Task** | Regression (predicting PCE %) |
| **Algorithm** | *[e.g., XGBoost / Random Forest / Gradient Boosting]* |
| **Training Data** | *[e.g., N samples from literature + DFT datasets]* |
| **Features** | 4 (band gap, defect density, trap depth, grain size) |
| **Target** | Power Conversion Efficiency (PCE, %) |
| **Performance** | R² = *[0.XX]*, RMSE = *[X.XX] %*, MAE = *[X.XX] %* |

### Why this approach?

- **Interpretable** — feature importances reveal which defects dominate efficiency loss
- **Fast inference** — predicts in milliseconds vs. hours for DFT
- **Extensible** — can be retrained as new perovskite datasets become available

---

## 🚀 Live Demo

👉 **Try it on Hugging Face Spaces:** [https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE)

Enter the four defect properties, and the model returns the predicted efficiency along with confidence intervals.

### Example Prediction

| Input | Value |
|---|---|
| Band Gap | 1.55 eV |
| Defect Density | 1 × 10¹⁵ cm⁻³ |
| Trap Depth | 0.10 eV |
| Grain Size | 500 nm |

**→ Predicted PCE: ~22.4 %**

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — model training & preprocessing
- **XGBoost / LightGBM** — gradient boosting
- **pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualization
- **Streamlit** — interactive web interface
- **Hugging Face Spaces** — deployment

---

## 📂 Project Structure
