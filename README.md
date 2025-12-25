## Introduction

This repository provides a minimal, didactic framework for studying **unstructured weight pruning** in small Transformer language models under strict hardware constraints. Instead of relying on full text generation or perplexity — which are impractical when models cannot be fully loaded into memory — the project focuses on two complementary axes:

- **Behavioural analysis:** token-level *top-1 agreement* between the dense model and its pruned variants, quantifying how pruning alters predictions.
- **Structural analysis:** per-layer sparsity, parameter distribution, and theoretical FLOPs reduction, including a comparison between dense, masked, and CSR-executed layers.

Three model variants are explored:
- **Dense baseline**  
- **Masked pruning (30%)** — unstructured magnitude pruning, weights set to zero but tensors remain dense  
- **CSR pruning (30%)** — same pruning, with selected linear layers converted to **Compressed Sparse Row (CSR)** format to reduce memory footprint and arithmetic cost

The repository notebooks illustrate the full workflow:  
**load model → prune → convert to CSR → evaluate top-1 stability → inspect structural effects.**  
This makes the project a compact, reproducible environment for understanding how sparsification affects both the behaviour and internal structure of Transformer LMs.

---

## Quick start (Google Colab, T4 GPU)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TaharHERRI/Edu-Sparsify-LLMs/blob/main/notebooks/Global_Behavior_&_Similarity.ipynb)   Notebook 1 - Global Behavior & Similarity

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TaharHERRI/Edu-Sparsify-LLMs/blob/main/notebooks/Structural_Layer_by_Layer_Analysis.ipynb)   Notebook 2 - Structural Layer by Layer Analysis

1. **Set the runtime to T4 GPU**  
   In Colab: **Runtime → Change runtime type → Hardware accelerator: GPU → GPU type: T4**.

2. **Clone the repo into `/content` using the terminal**  
   In Colab, open a terminal and run:

   ```bash
   git clone https://github.com/TaharHERRI/sparsify-min.git /content/sparsify-min
   ```

3. **Add a `%cd` cell at the top of the notebook (before any imports)**  
   Open `notebooks/S1_minimal.ipynb` and insert this as the **first** cell:

   ```python
   %cd /content/sparsify-min/notebooks
   ```

Then run the notebook cells in order.

> **Prefer not to use Colab?**  
You can run the notebook locally with **Jupyter Lab** (see “Local setup” below).

---

## Local setup (optional, Jupyter Lab)

```bash
python3 -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

# Launch Jupyter and open the notebook:
jupyter lab --no-browser --ServerApp.use_redirect_file=False # Launch without opening a browser; copy the URL printed in your terminal into your local browser
# then open: notebooks/S1_minimal.ipynb
```

A CUDA-capable GPU is helpful but not required to run the demo.

---

## Repository layout

```
Repository layout
├── notebooks/
├── src/
│   ├── eval/                        
│   ├── pruning/                     
│   └── wrappers/                    
├── docs/                            
├── requirements.txt
└── README.md
```

---

## Notes

* **Masked pruning vs. execution:** magnitude pruning sets a fraction of weights to zero but the layers remain dense; computation still uses standard dense GEMM. This mode is useful to study behavioural and structural effects of sparsity without introducing changes in the execution kernels.

* **CSR path:** after pruning, selected linear layers are converted to a **Compressed Sparse Row (CSR)** representation and wrapped with a sparse-aware forward operator. This reduces stored parameters and theoretical FLOPs. Actual sparse acceleration depends on backend support (e.g., MKL on CPU, cuSPARSE on GPU); on unsupported environments, the fallback is a Python-level sparse matmul.

* **Device selection:** notebooks automatically choose GPU (CUDA) when available; otherwise, all steps run on CPU. CSR execution is still functional on CPU-only setups, but without GPU-level sparse kernel acceleration.

* **Reproducibility:** random seeds are set consistently, though exact numerical outputs may vary across hardware, driver versions, and linear algebra libraries.

