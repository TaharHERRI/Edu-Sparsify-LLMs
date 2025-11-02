# Sparsify-LLM (minimal)

A small, readable baseline to demonstrate:
1) Dense baseline
2) Unstructured masked pruning (no speedup)
3) CSR execution (real sparse kernels)

Perplexity is computed robustly (pads ignored, FP32). We prune only safe Linear
layers (skip `lm_head`) and use moderate sparsity for a sane demo.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate #(on Windows) .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab --no-browser --ServerApp.use_redirect_file=False
# Open notebooks/S1_minimal.ipynb
```
