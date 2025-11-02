import math, torch

def params_size_and_sparsity(model):
    """Parameter-only size (MB) and sparsity (ratio of zeros). Safe with CSR buffers."""
    nz = 0; tot = 0; bytes_ = 0
    with torch.no_grad():
        for p in model.parameters():
            t = p.detach()
            tot += t.numel()
            nz  += int((t != 0).sum().item())
            bytes_ += t.numel() * t.element_size()
    size_mb = bytes_ / (1024 * 1024)
    sparsity = 0.0 if tot == 0 else 1.0 - (nz / tot)
    return {"nonzero": nz, "total": tot, "sparsity": sparsity, "size_mb": size_mb}

@torch.no_grad()
def eval_ppl_causal(model, tokenizer, texts, device, max_length=256, add_eos=True):
    """Robust perplexity for causal LMs (FP32, pads ignored via -100 labels)."""
    enc = tokenizer(
        texts if add_eos else [t.rstrip(tokenizer.eos_token) for t in texts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attn == 0] = -100  # ignore pad tokens

    if device == "cuda":
        with torch.autocast("cuda", enabled=False):
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    else:
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)

    loss = out.loss.detach().float()
    return math.exp(loss.item())
