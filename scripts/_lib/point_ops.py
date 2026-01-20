from __future__ import annotations

import torch


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return kNN indices for each point. x is (B, C, N)."""
    if x.ndim != 3:
        raise ValueError(f"Expected x shape (B,C,N), got {tuple(x.shape)}")
    b, _c, n = x.shape
    if n <= 1:
        return torch.zeros((b, n, 0), dtype=torch.long, device=x.device)
    kk = max(1, min(int(k), n - 1))

    # Indices only; gradients through kNN are not meaningful and explode memory.
    with torch.no_grad():
        xt = x.transpose(2, 1).contiguous()  # (B, N, C)
        xt_dist = xt
        if xt_dist.device.type == "cuda" and xt_dist.dtype == torch.float32:
            xt_dist = xt_dist.to(dtype=torch.float16)
        # Pairwise squared distance: ||a-b||^2 = ||a||^2 - 2a·b + ||b||^2
        xx = torch.sum(xt_dist * xt_dist, dim=2, keepdim=True)  # (B, N, 1)
        inner = -2.0 * torch.matmul(xt_dist, xt_dist.transpose(2, 1))  # (B, N, N)
        dist = xx + inner + xx.transpose(2, 1)  # (B, N, N)

        idx = dist.topk(k=kk + 1, dim=-1, largest=False).indices[:, :, 1:]  # exclude self
        return idx  # (B, N, k)


def get_graph_feature(x: torch.Tensor, *, k: int) -> torch.Tensor:
    """Construct EdgeConv features. Returns (B, 2C, N, k)."""
    if x.ndim != 3:
        raise ValueError(f"Expected x shape (B,C,N), got {tuple(x.shape)}")
    b, c, n = x.shape
    idx = knn(x, k=k)  # (B, N, k)
    if idx.numel() == 0:
        return x.new_zeros((b, 2 * c, n, 0))

    idx_base = (torch.arange(0, b, device=x.device).view(-1, 1, 1) * n).long()
    idx2 = (idx + idx_base).reshape(-1)  # (B*N*k,)

    xt = x.transpose(2, 1).contiguous()  # (B, N, C)
    neighbors = xt.reshape(b * n, c)[idx2, :].reshape(b, n, -1, c)  # (B, N, k, C)
    center = xt.reshape(b, n, 1, c).expand(-1, -1, neighbors.shape[2], -1)
    edge = torch.cat([neighbors - center, center], dim=3).permute(0, 3, 1, 2).contiguous()
    return edge  # (B, 2C, N, k)

