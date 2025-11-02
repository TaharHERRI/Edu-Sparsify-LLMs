import torch
from torch.nn.utils import prune

def select_prunable_linears(model, blacklist=("lm_head",)):
    """Return Linear modules that are safe to prune (skip names with blacklist)."""
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(b in name for b in blacklist):
                continue
            prunable.append(module)
    return prunable

@torch.no_grad()
def apply_global_magnitude_pruning_cpu_safe(modules, amount: float, param_name: str = "weight"):
    """Global unstructured pruning without GPU OOM."""
    flat_scores = []; shapes = []; refs = []; total = 0
    for m in modules:
        if not hasattr(m, param_name): continue
        t = getattr(m, param_name).detach()
        shapes.append(t.shape); refs.append((m, t.device))
        s = t.abs().to("cpu", copy=True).view(-1)
        flat_scores.append(s); total += s.numel()
    if total == 0: return
    scores_cpu = torch.cat(flat_scores, dim=0)
    k = int(amount * total)
    if k <= 0:      thr = float("-inf")
    elif k >= total: thr = float("inf")
    else:           thr = torch.kthvalue(scores_cpu, k).values.item()
    off = 0
    for (m, dev), shape in zip(refs, shapes):
        n = shape.numel()
        local = (scores_cpu[off:off+n].view(shape) > thr).to(dev)
        off += n
        prune.CustomFromMask.apply(m, param_name, local)
