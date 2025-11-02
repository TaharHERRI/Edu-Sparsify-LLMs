import torch

class LinearCSRForward(torch.nn.Module):
    """Minimal CSR forward wrapper: y = x @ W^T + b, with W stored as CSR."""
    def __init__(self, W_dense: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("W_csr", W_dense.to_sparse_csr())
        self.register_buffer("bias", bias if bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.W_csr.transpose(0,1).to_dense())
        if self.bias is not None:
            out = out + self.bias
        return out
