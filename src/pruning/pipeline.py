import torch, warnings
from torch.nn.utils import prune

def freeze_pruning_(modules, param_name: str = "weight"):
    """Permanently apply masks and remove all pruning reparams + hooks."""
    for m in modules:
        # Pruning adds <param>_orig and <param>_mask and registers a forward_pre_hook.
        # prune.remove() multiplies weight by mask, restores plain parameter, and unregisters the hook.
        if hasattr(m, f"{param_name}_orig"):
            prune.remove(m, param_name)

def convert_linear_weights_to_csr_(modules, param_name: str = "weight"):
    """Store a CSR buffer inside each module for demo (only used by a wrapper)."""
    for m in modules:
        W = getattr(m, param_name, None)
        if not isinstance(W, torch.Tensor): continue
        try:
            m.register_buffer("_W_csr", W.to_sparse_csr())
        except Exception as e:
            warnings.warn(f"CSR conversion failed for {m}: {e}")

