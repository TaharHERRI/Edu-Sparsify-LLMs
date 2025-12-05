import torch
import torch.nn as nn


class LinearCSRForward(nn.Module):
    """
    Couche linéaire avec poids stocké en CSR.

    - Forward : (x @ W^T) + b, comme nn.Linear
    - Stats : garde nnz, total, sparsité, shape d'origine
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()

        assert weight.dim() == 2, "LinearCSRForward attend un poids [out_features, in_features]"
        out_features, in_features = weight.shape

        # --- Stats denses AVANT compression ---
        dense_total = out_features * in_features
        dense_nnz = int((weight != 0).sum().item())

        self.meta_out_features = out_features
        self.meta_in_features = in_features
        self.meta_total_params = int(dense_total)
        self.meta_nnz = int(dense_nnz)
        self.meta_sparsity = 1.0 - self.meta_nnz / self.meta_total_params

        self.register_buffer(
            "meta_dense_shape",
            torch.tensor([out_features, in_features], dtype=torch.long),
            persistent=False,
        )

        # --- Poids en CSR pour le forward ---
        weight_csr = weight.to_sparse_csr()
        self.register_buffer("weight_csr", weight_csr, persistent=False)

        # Biais (on peut garder Parameter, même si on n'entraîne pas)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        out: (..., out_features)
        """
        W = self.weight_csr

        # S'assurer que tout est sur le même device
        if W.device != x.device:
            W = W.to(x.device)

        # Aplatir sur la dernière dimension comme nn.Linear
        orig_shape = x.shape                       # (..., in_features)
        in_features = self.meta_in_features
        out_features = self.meta_out_features

        x_flat = x.view(-1, in_features)           # (N, in_features)

        # W: (out,in), x_flat.T: (in,N) -> (out,N) -> (N,out)
        out_flat = torch.matmul(W, x_flat.T).T     # (N, out_features)

        if self.bias is not None:
            out_flat = out_flat + self.bias

        out = out_flat.view(*orig_shape[:-1], out_features)
        return out

    # Petites helpers pour l'analyse
    @property
    def dense_total_params(self):
        return self.meta_total_params

    @property
    def dense_nnz(self):
        return self.meta_nnz

    @property
    def dense_sparsity(self):
        return self.meta_sparsity
