# IO-GNN - Graph-constrained LSTM for Inter-Industry Forecasting

Minimal, research-grade implementation of our **ChebDirConv + GC-LSTM**
architecture for simultaneous prediction of  

* **Z(t)** — inter-industry transaction matrix  
* **X(t)** — sector-level total output  

given a moving input window of technical-coefficient graphs **Af(t-4 … t-1)**.

> **Status** : under active writing - code is public for transparency & easy Colab
> access.  
> **Data / trained weights are _not_ included.**

---

## Quick start

```bash
git clone https://github.com/<user>/io-gnn.git
cd io-gnn
pip install -r requirements.txt
python main.py                      # runs λ×β grid + early-stopping
