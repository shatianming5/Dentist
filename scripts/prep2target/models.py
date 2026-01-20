from __future__ import annotations

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        ld = int(latent_dim)
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, ld, 1),
            nn.BatchNorm1d(ld),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected (B,N,3), got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        feat = self.net(x)
        return feat.max(dim=2).values


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, n_points: int) -> None:
        super().__init__()
        ld = int(latent_dim)
        n = int(n_points)
        self.n_points = n
        self.net = nn.Sequential(
            nn.Linear(ld, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        return x.view(z.shape[0], self.n_points, 3)


class Prep2TargetNet(nn.Module):
    def __init__(self, latent_dim: int, n_points: int) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim=int(latent_dim))
        self.decoder = MLPDecoder(latent_dim=int(latent_dim), n_points=int(n_points))

    def forward(self, prep: torch.Tensor) -> torch.Tensor:
        z = self.encoder(prep)
        return self.decoder(z)


class Prep2TargetLabelNet(nn.Module):
    def __init__(self, *, latent_dim: int, n_points: int, num_labels: int) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim=int(latent_dim))
        self.decoder = MLPDecoder(latent_dim=int(latent_dim), n_points=int(n_points))
        self.label_emb = nn.Embedding(int(num_labels), int(latent_dim))

    def forward(self, prep: torch.Tensor, label_id: torch.Tensor) -> torch.Tensor:
        z = self.encoder(prep)
        if label_id.ndim == 0:
            label_id = label_id.view(1)
        z = z + self.label_emb(label_id.to(dtype=torch.long, device=z.device))
        return self.decoder(z)


class ConstraintAuxHead(nn.Module):
    def __init__(self, *, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        ld = int(latent_dim)
        hd = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(ld, hd),
            nn.ReLU(inplace=True),
            nn.Linear(hd, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def forward_pred_and_latent(
    model: nn.Module, prep: torch.Tensor, label_id: torch.Tensor, *, cond_label: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(model, "encoder") or not hasattr(model, "decoder"):
        raise ValueError("Expected model to have .encoder and .decoder")
    z = model.encoder(prep)
    if cond_label:
        if not hasattr(model, "label_emb"):
            raise ValueError("cond_label=True but model has no label_emb")
        if label_id.ndim == 0:
            label_id = label_id.view(1)
        z = z + model.label_emb(label_id.to(dtype=torch.long, device=z.device))
    pred = model.decoder(z)
    return pred, z

