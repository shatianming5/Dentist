from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class RawPrep2TargetDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            dict[str, Any],
        ]
    ]
):
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        data_root: Path,
        label_to_id: dict[str, int] | None = None,
    ) -> None:
        self.rows = rows
        self.data_root = data_root
        self.label_to_id = label_to_id or {}

    def __len__(self) -> int:  # noqa: D401
        return len(self.rows)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        r = self.rows[int(idx)]
        npz_path = self.data_root / str(r["sample_npz"])
        with np.load(npz_path) as data:
            prep = np.asarray(data["prep_points"], dtype=np.float32)
            tgt = np.asarray(data["target_points"], dtype=np.float32)
            margin = np.asarray(data["margin_points"], dtype=np.float32)
            opp = np.asarray(data["opp_points"], dtype=np.float32)
            centroid = np.asarray(data["centroid"], dtype=np.float32).reshape(3)
            scale = np.asarray(data["scale"], dtype=np.float32).reshape(())
            rmat = np.asarray(data["R"], dtype=np.float32).reshape(3, 3)

        label = str(r.get("label") or "")
        label_id = int(self.label_to_id.get(label, 0))

        return (
            torch.from_numpy(prep),
            torch.from_numpy(tgt),
            torch.from_numpy(margin),
            torch.from_numpy(opp),
            torch.from_numpy(centroid),
            torch.from_numpy(scale),
            torch.from_numpy(rmat),
            torch.tensor(label_id, dtype=torch.long),
            {
                "case_key": str(r.get("case_key") or ""),
                "label": str(r.get("label") or ""),
                "sample_npz": str(r.get("sample_npz") or ""),
                "split": str(r.get("split") or ""),
            },
        )

