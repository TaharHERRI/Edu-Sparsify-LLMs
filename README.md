# sparsify-min

Minimal, didactic repo that shows three execution modes for small causal LMs:

1. **Dense baseline**
2. **Unstructured (masked) pruning** — sets weights to zero but still runs dense kernels
3. **CSR execution** — converts selected linear layers to a sparse format to use sparse kernels when available (e.g., MKL on CPU / cuSPARSE on GPU)

The main notebook demonstrates the end-to-end flow: load model, (optionally) prune, convert to CSR, and run inference/evaluation.

---

## Quick start (Google Colab, T4 GPU)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TaharHERRI/sparsify-min/blob/main/notebooks/S1_minimal.ipynb)

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
notebooks/
  S1_minimal.ipynb      # Main demo: dense → masked → CSR
src/
  eval/                 # utilities (metrics, latency helpers, plotting hooks)
  pruning/              # pruning policies + pipeline (freeze/convert)
  wrappers/             # CSR execution (e.g., LinearCSRForward)
requirements.txt
README.md
```

---

## Notes

* **Masked pruning vs. execution:** masking zeros weights in place but continues to use dense ops; it is primarily useful to illustrate sparsity without changing kernels.
* **CSR path:** selected linear layers are converted to CSR and swapped with a sparse-aware forward; when the backend supports it, sparse kernels may be used (e.g., MKL/cuSPARSE).
* **Device selection:** the notebook auto-selects CUDA if available; otherwise it runs on CPU.
* **Reproducibility:** seeds are set in code where relevant; exact outputs may still vary by device/driver/library versions.
