# IO-GNN  
**Graph-constrained LSTM for Inter-Industry Forecasting based on the Input-Output Table**

Minimal, research-grade implementation of our `ChebDirConv + GC-LSTM` model for forecasting industrial structure via Graph Neural Networks and PINN-style constraints.

---

## 🔍 Overview

This repository proposes a novel framework to predict:

- **Z(t)**: Inter-industry transaction matrix (log-transformed)
- **X(t)**: Sector-level total output  
- **Subject**: Korea / China / Japan  
- **Model**: Directed Chebyshev + LSTM with Input-Output PINN loss

---

## 📂 Project Structure

```
IO_GNN/
│
├── config.py          # Configuration for training & model
├── data.py            # Data loading & preprocessing
├── losses.py          # PINN-based economic constraint losses
├── model.py           # ChebDirConv + GC-LSTM implementation
├── trainer.py         # Model training logic
├── utils.py           # Utility functions (dumping results)
├── main.py            # Entry point for single run
├── sweep.py           # Grid search over λ, β
└── README.md
```

---

## 📁 Input File Format

You need CSV files with names like:

```
data/
│
├── X_1.csv       # Node features (Imports, Exports, Final Demand, Value Added, Total)
├── Af_1.csv      # Edge weights (Technical coefficients) for training
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

### 1. Run single training

```bash
python main.py
```

### 2. Run grid search with multiple λ/β/seeds (takes time!)

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
                X_000.csv
```

---

## 📊 Sample Output

| Epoch | SMAPE ↓ | RMSE ↓ | CVR ↓ |
|-------|---------|--------|--------|
| 30    | 43.23%  | 4553   | 8.72%  |
| 60    | 25.91%  | 3728   | 3.50%  |
| 120   | 28.10%  | 2920   | 3.41%  |
| 300   | 23.94%  | 2153   | 3.53%  |

---

## 🧠 Notes

- λ controls the strength of economic balance constraints (PINN loss)
- β controls the weight of node prediction loss (X̂ vs X)
- CVR (%) measures how well the predicted outputs meet input-output accounting identities

---

## 📌 Citation

📄 *Will be added upon paper submission*

---

## 🔒 License

**Do not use without permission.** This codebase is under active research and not yet published. For academic collaboration or permission to reuse, contact the author.

---

## ✉️ Contact

Feel free to reach out for questions or collaboration:

- 🧑‍💻 GitHub: [@MinQ09](https://github.com/MinQ09)
- ✉️ Email: kmingyu12@kaist.ac.kr
