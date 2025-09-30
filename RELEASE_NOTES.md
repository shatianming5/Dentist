# Dentist 3D Toolkit – v0.1.0

This release establishes an end-to-end pipeline around your dental (upper/lower arch) 3D data and a small backend for inspection.

Highlights
- Data ingest
  - CloudCompare BIN v2 native probe/reader: `scripts/read_ccbin.py`, `scripts/ccbin_v2_reader.py`
  - Batch convert to PLY (via CloudCompare CLI): `tools/convert-all-bin-to-ply.ps1`
- Dataset prep
  - Export normalized NPZ (N×3) from BIN: `scripts/export_points_npz.py`
  - Split generator (train/val/test): `scripts/generate_splits.py`
- Self-supervised pretraining (lightweight, CPU-friendly)
  - SimCLR with PointNet encoder: `scripts/train_simclr_pointnet.py`
  - Linear probe (Upper vs Lower): `scripts/linear_probe_pointnet.py`
- Backend service (FastAPI)
  - Endpoints: `/health`, `/files/bin`, `/points/bin/{filename}?peek=N`
  - Launch: `uvicorn backend.app:app --host 127.0.0.1 --port 8003`

What this version is for
- Quickly explore and read your dental BIN files, preview points (+RGB if available).
- Standardize dataset into NPZ (centered, robust unit-sphere normalization), and produce train/val/test lists.
- Run a minimal self-supervised loop to verify representation learning on your data, and a simple linear probe for an immediate quality check.
- Provide a small backend so a front-end can list files and preview sample points.

Quick start
1) Env
- `conda create -n dentist-3d python=3.9 -y && conda run -n dentist-3d python -m pip install -r requirements.txt`
- (CPU torch) `conda run -n dentist-3d python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

2) Export + splits
- `conda run -n dentist-3d python scripts/export_points_npz.py --src data --dst data_npz --points 2048`
- `conda run -n dentist-3d python scripts/generate_splits.py --src data_npz --ext .npz --out splits --val-ratio 0.1 --test-ratio 0.1`

3) Pretrain + probe
- `conda run -n dentist-3d python scripts/train_simclr_pointnet.py --root data_npz --train_list splits/train.txt --val_list splits/val.txt --points 2048 --epochs 20 --batch_size 16 --out outputs/simclr_pointnet`
- `conda run -n dentist-3d python scripts/linear_probe_pointnet.py --root data_npz --train_list splits/train.txt --val_list splits/val.txt --points 2048 --epochs 20 --batch_size 16 --ckpt outputs/simclr_pointnet/ckpt_best.pth --out outputs/linear_probe`

4) Backend
- `conda run -n dentist-3d python -m uvicorn backend.app:app --host 127.0.0.1 --port 8003`
- Open `http://127.0.0.1:8003/docs`

Notes
- For full-fidelity parsing/conversion, prefer CloudCompare PLY export.
- SimCLR pipeline is a lightweight baseline; we can integrate PointMAE/PointBERT/PointContrast next.
- NPZ export now filters non-finite points and uses robust scaling to avoid overflow/NaNs.

