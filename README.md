# IO-GNN  
**Directional Graph-constrained LSTM for Inter-Industry Forecasting based on the Input-Output Table**

Minimal, research-grade implementation of our `DirMPNN + GraphLSTMCell` model for forecasting industrial structure via Graph Neural Networks with **PINN-style constraints**.

---

## 🔍 Overview

This repository proposes a novel framework to predict:

- **Z(t)**: Inter-industry transaction matrix (log-transformed)  
- **VA(t)**: Value added at sector level  
- **X(t)**: Sector-level total output  
- **Scope**: Korea / China / Japan  
- **Model**: Directional Message Passing + Graph LSTM with Input-Output PINN loss  

### Key Features (updated)

- **Directional message passing (`DirMPNN`)** with forward/backward edges.  
- **GraphLSTMCell** with improved forget-gate bias initialization (+1).  
- **Configurable options**:  
  - `use_edge_weight`, `use_bwd_weights`  
  - `compute_attention` (analysis-only attention scores)  
  - `alpha_mode` (`scalar` or `channel` mixing of fwd/bwd)  
  - `va_nonneg` (Softplus head for non-negative VA).  
- **PINN constraints** to enforce IO accounting identities (input=output, etc.).  

---

## 📂 Project Structure

```
IO_GNN/
│
├── config.py          # Config with new flags (edge usage, attention, alpha_mode, va_nonneg)
├── data.py            # Data loading & preprocessing
├── losses.py          # PINN-based economic constraint losses
├── model.py           # DirMPNN + GraphLSTMCell implementation
├── trainer.py         # Model training logic
├── utils.py           # Utility functions (dumping results)
├── main.py            # Entry point for single run
├── sweep.py           # Grid search over λ, β
└── README.md
```

---

## 📁 Input File Format

Expected CSV files (log-scale for edges recommended):

```
data/
│
├── X_1.csv       # Node features (Imports, Exports, Final Demand, Value Added, Total)
├── Af_1.csv      # Edge weights (Technical coefficients)
├── Zf_1.csv      # Target edge matrix (Inter-industry transactions)
```

### Example: `X_1.csv`

| Imports | Exports | Final_Demand | Value_Added | Total |
|---------|---------|---------------|-------------|--------|
| 123456  | 7890    | 56789         | 23456       | 123456 |

---

## ⚙️ Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If you're on Colab, install `torch-geometric` with:

```python
!pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

## 🚀 How to Run

### 1. Single training run

```bash
python main.py
```

### 2. Grid search over λ/β/seeds

```bash
python sweep.py
```

Results are saved in:

```
<out_dir>/
    seed_<n>/
        lam_<λ>_beta_<β>/
            model.pth
            train_history.json
            csv_pred/
                pred_Z_000.csv, ...
                pred_VA_000.csv
                X_000.csv
```

---

## 📊 Sample Output

| Epoch | SMAPE(Z) ↓ | RMSE ↓ | CVR ↓ |
|-------|------------|--------|--------|
| 30    | 43.2%      | 4553   | 8.72%  |
| 60    | 25.9%      | 3728   | 3.50%  |
| 120   | 28.1%      | 2920   | 3.41%  |
| 300   | 23.9%      | 2153   | 3.53%  |

---

## 🧠 Notes

- **λ** controls the strength of PINN loss (economic balance).  
- **β** controls the node prediction loss (X̂ vs X).  
- **CVR (%)**: Input-Output Consistency score.  
- **Flags** in `config.py`:  
  - `use_edge_weight` = include edge_attr in messages.  
  - `use_bwd_weights` = use backward edge_attr_bwd.  
  - `compute_attention` = toggle attention score computation (analysis only).  
  - `alpha_mode` = forward/backward mixing rule.  
  - `va_nonneg` = enforce non-negative VA prediction.  

---

## 📌 Citation

📄 *To be added upon paper submission.*

---

## 🔒 License

**Do not use without permission.** This codebase is under active research and not yet published. For academic collaboration or permission to reuse, contact the author.

---

## ✉️ Contact

- 🧑‍💻 GitHub: [@MinQ09](https://github.com/MinQ09)  
- ✉️ Email: kmingyu12@kaist.ac.kr  
