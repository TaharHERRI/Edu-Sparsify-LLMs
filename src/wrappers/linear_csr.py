import torch

class LinearCSRForward(torch.nn.Module):
    """Minimal CSR forward wrapper: y = x @ W^T + b, with W stored as CSR."""
    def __init__(self, W_dense: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("W_csr", W_dense.to_sparse_csr())
        self.register_buffer("bias", bias if bias is not None else None)

    # W_csr: [out_features, in_features] (CSR sparse)
    # x:     [batch, in_features]       (dense)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.W_csr  # rester sparse
        try:
            # S @ D -> D  (exploite MKL/cuSPARSE selon device/build)
            out = torch.matmul(W, x.T).T               # [out,b] -> [b,out]
        except RuntimeError:
            # Fallback si l’opération sparse n’est pas dispo
            out = x @ W.to_dense().T
        if self.bias is not None:
            out = out + self.bias
        return out
