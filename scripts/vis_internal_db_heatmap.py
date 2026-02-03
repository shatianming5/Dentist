#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from internal_db.heatmap_ops import (
    HeatField,
    compute_custom_scalar_on_vertices,
    map_point_scalar_to_vertices,
    normalize01,
    sample_points_on_polydata,
    smooth_vertex_scalar_knn,
)
from internal_db.icp import icp_align
from internal_db.mesh_io import (
    OptionalDependencyError,
    apply_alignment,
    bounds_center_radius,
    compute_case_alignment,
    infer_camera,
    load_stl_as_polydata,
    prepare_polydata,
)
from internal_db.model_infer import (
    heat_from_probs,
    infer_raw_seg_probs,
    load_raw_cls_model,
    load_raw_seg_model,
    raw_cls_point_saliency,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _pick_case(records: list[dict[str, Any]], *, case_uid: str | None, query: str | None, pick_latest: bool) -> dict[str, Any]:
    if case_uid:
        for r in records:
            if str(r.get("case_uid", "")) == str(case_uid):
                return r
        raise SystemExit(f"case_uid not found: {case_uid}")
    if query:
        q = str(query).strip()
        for r in records:
            if q in str(r.get("source_relpath", "")):
                return r
        for r in records:
            assets = r.get("assets", {}) or {}
            for k in ("upper_stl", "lower_stl", "bite_stl"):
                cands = ((assets.get(k, {}) or {}).get("candidates")) or []
                for c in cands:
                    if q in str((c or {}).get("relpath", "")):
                        return r
        raise SystemExit(f"No record matches query: {query!r}")
    if pick_latest:

        def _key(r: dict[str, Any]) -> float:
            try:
                return float(r.get("mtime_max", 0.0))
            except Exception:
                return 0.0

        return max(records, key=_key)
    raise SystemExit("Must provide one of: --case-uid, --query, --pick-latest")


def _resolve_best_path(record: dict[str, Any], *, root: Path, key: str) -> Path | None:
    assets = record.get("assets", {}) or {}
    best = ((assets.get(key, {}) or {}).get("best")) or None
    if not best:
        return None
    relpath = str(best.get("relpath", "")).strip()
    if not relpath:
        return None
    p = Path(relpath)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _hex_to_rgb(s: str) -> tuple[int, int, int] | None:
    v = str(s).strip().lstrip("#")
    if len(v) == 3:
        v = "".join([c * 2 for c in v])
    if len(v) != 6:
        return None
    try:
        r = int(v[0:2], 16)
        g = int(v[2:4], 16)
        b = int(v[4:6], 16)
    except Exception:
        return None
    return r, g, b


def _blended_colorscale(*, base_hex: str, cmap: str, cut: float, cmap_start: float) -> Any:
    """Blend a base (tooth) color into a plotly colorscale for readability.

    For many heatmaps, the low end of common colormaps is near-black which
    makes the tooth disappear in screenshots. This function keeps low heat
    values close to `base_hex` while preserving the chosen colormap for
    mid/high values.
    """
    try:
        import plotly.colors as pc
    except Exception:
        return str(cmap)

    name = str(cmap).strip()
    if not name:
        return str(cmap)

    try:
        orig = pc.get_colorscale(name)
    except Exception:
        return name

    c = float(cut)
    c = max(0.0, min(0.6, c))
    if c <= 1e-6:
        return orig

    s = float(cmap_start)
    s = max(0.0, min(1.0, s))

    # Sample the colormap from [s, 1] and remap to [c, 1].
    n = 10
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    sample_pts = (s + (1.0 - s) * ts).astype(float).tolist()
    cols = pc.sample_colorscale(orig, sample_pts)

    base_end = max(0.0, c - 1e-6)
    scale: list[list[Any]] = [[0.0, str(base_hex)], [base_end, str(base_hex)]]
    for idx, col in enumerate(cols):
        pos = c + (1.0 - c) * (float(idx) / float(max(1, n - 1)))
        scale.append([pos, col])
    return scale


def _crop_png(path: Path, *, bg_hex: str, threshold: int, pad: int) -> None:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im)

    bg = _hex_to_rgb(bg_hex) or tuple(int(x) for x in arr[0, 0, :3])
    bg_arr = np.asarray(bg, dtype=np.int16)
    diff = np.abs(arr.astype(np.int16) - bg_arr[None, None, :])
    mask = diff.max(axis=2) > int(threshold)
    if not bool(mask.any()):
        return
    ys, xs = np.where(mask)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1

    p = int(pad)
    y0 = max(0, y0 - p)
    x0 = max(0, x0 - p)
    y1 = min(arr.shape[0], y1 + p)
    x1 = min(arr.shape[1], x1 + p)

    with Image.open(path) as im2:
        im2 = im2.convert("RGB")
        cropped = im2.crop((x0, y0, x1, y1))
        cropped.save(path)


def _ensure_kaleido_chrome() -> None:
    if os.environ.get("BROWSER_PATH"):
        return
    chrome_candidates = [
        Path.home() / ".local/share/plotly/chrome-linux64/chrome",
        Path.home() / ".local/share/kaleido/chrome-linux64/chrome",
    ]
    for c in chrome_candidates:
        if c.exists() and os.access(c, os.X_OK):
            os.environ["BROWSER_PATH"] = str(c)
            return
    for exe in ["google-chrome", "google-chrome-stable", "chrome", "chromium", "chromium-browser"]:
        p = shutil.which(exe)
        if p:
            os.environ["BROWSER_PATH"] = p
            return


def _parse_quantiles(s: str) -> tuple[float, float] | None:
    v = str(s).strip().lower()
    if v in {"", "none", "off"}:
        return None
    if "," not in v:
        raise ValueError("quantiles must be 'qlo,qhi' or 'none'")
    a, b = v.split(",", 1)
    qlo = float(a)
    qhi = float(b)
    qlo = max(0.0, min(1.0, qlo))
    qhi = max(0.0, min(1.0, qhi))
    if qhi < qlo:
        qlo, qhi = qhi, qlo
    return qlo, qhi


def _vis_scale_array(x01: np.ndarray, *, lo: float, hi: float, gamma: float) -> np.ndarray:
    x = np.asarray(x01, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    if float(hi - lo) > 1e-8:
        x = (x - float(lo)) / float(hi - lo)
    x = np.clip(x, 0.0, 1.0)
    g = float(gamma)
    if abs(g - 1.0) > 1e-9:
        x = np.power(x, g, dtype=np.float32)
    return x.astype(np.float32, copy=False)


def _poly_to_plotly_triangles(poly: Any, *, center: np.ndarray) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int]]:
    pts = np.asarray(poly.points, dtype=np.float32) - center.astype(np.float32)
    faces = np.asarray(poly.faces)
    if faces.size % 4 != 0:
        raise ValueError("Expected triangular faces array in pyvista PolyData")
    f = faces.reshape(-1, 4)
    if not (f[:, 0] == 3).all():
        raise ValueError("Non-triangle faces found (call triangulate() before exporting)")
    tri = f[:, 1:4].astype(int)
    x = pts[:, 0].astype(float).tolist()
    y = pts[:, 1].astype(float).tolist()
    z = pts[:, 2].astype(float).tolist()
    i = tri[:, 0].tolist()
    j = tri[:, 1].tolist()
    k = tri[:, 2].tolist()
    return x, y, z, i, j, k


def _poly_to_plotly_points(poly: Any, *, center: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    pts = np.asarray(poly.points, dtype=np.float32) - center.astype(np.float32)
    mp = int(max_points)
    if mp > 0 and pts.shape[0] > mp:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(int(pts.shape[0]), size=mp, replace=False)
        pts = pts[idx]
    return pts.astype(np.float32, copy=False)


def _build_camera(bounds: tuple[float, float, float, float, float, float], *, view: str, eye_mult: float) -> tuple[dict[str, float], dict[str, float]]:
    center, radius = bounds_center_radius(bounds)
    cam_pos, focal, up = infer_camera(center, radius, view=view)
    rad = float(radius) if float(radius) > 1e-6 else 1.0
    e = float(eye_mult)
    eye = {
        "x": float(cam_pos[0] - focal[0]) / rad * e,
        "y": float(cam_pos[1] - focal[1]) / rad * e,
        "z": float(cam_pos[2] - focal[2]) / rad * e,
    }
    upv = {"x": float(up[0]), "y": float(up[1]), "z": float(up[2])}
    return eye, upv


def _load_external_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (points, heat). Heat should be 0..1 (will be normalized if needed)."""
    p = path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Missing external npz: {p}")
    with np.load(p) as z:
        if "points" in z.files:
            pts = np.asarray(z["points"], dtype=np.float32)
        elif "xyz" in z.files:
            pts = np.asarray(z["xyz"], dtype=np.float32)
        else:
            raise ValueError(f"{p}: expected 'points' or 'xyz' in npz, got keys={list(z.files)}")

        if "heat" in z.files:
            heat = np.asarray(z["heat"], dtype=np.float32).reshape(-1)
        elif "scalar" in z.files:
            heat = np.asarray(z["scalar"], dtype=np.float32).reshape(-1)
        else:
            raise ValueError(f"{p}: expected 'heat' or 'scalar' in npz, got keys={list(z.files)}")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{p}: invalid points shape {tuple(pts.shape)}")
    if heat.shape[0] != pts.shape[0]:
        raise ValueError(f"{p}: heat length {heat.shape[0]} != points {pts.shape[0]}")
    return pts.astype(np.float32, copy=False), normalize01(heat)


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay model/custom heatmaps onto intraoral STL meshes and export HTML + screenshots.")
    ap.add_argument("--index", type=Path, default=Path("metadata/internal_db/index.jsonl"), help="Index JSONL file.")
    ap.add_argument("--root", type=Path, default=Path("downloads/体内数据库/体内数据库"), help="Root folder (for resolving relative paths).")

    sel = ap.add_argument_group("Selection")
    sel.add_argument("--case-uid", type=str, default="", help="Select by case_uid.")
    sel.add_argument("--query", type=str, default="", help="Substring match on source_relpath / asset paths.")
    sel.add_argument("--pick-latest", action="store_true", help="Pick record with max mtime_max.")

    rend = ap.add_argument_group("Rendering")
    rend.add_argument("--show", type=str, default="upper,lower", help="Comma list: upper,lower,bite")
    rend.add_argument("--align", type=str, default="pca", choices=["none", "center", "pca"])
    rend.add_argument("--align-from", type=str, default="upperlower", choices=["all", "upperlower", "upper", "lower"])
    rend.add_argument("--camera-from", type=str, default="upperlower", choices=["all", "upperlower", "upper", "lower"])
    rend.add_argument("--decimate", type=float, default=0.15, help="Mesh target reduction ratio in [0,1).")
    rend.add_argument("--smooth-iters", type=int, default=0)
    rend.add_argument("--theme", type=str, default="auto", choices=["auto", "light", "dark"])
    rend.add_argument("--background", type=str, default="#f7f7f7", help="Hex color, e.g. '#0b1020'.")
    rend.add_argument("--plotly-eye-mult", type=float, default=0.55, help="Smaller -> zoom in.")
    rend.add_argument("--cmap", type=str, default="magma", help="Plotly colorscale name (e.g., magma/viridis/turbo).")
    rend.add_argument("--render", type=str, default="both", choices=["mesh", "points", "both"], help="Render mesh heat, points overlay, or both.")
    rend.add_argument(
        "--mesh-heat-blend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Blend a tooth base color into the heat colormap so low-heat regions remain visible.",
    )
    rend.add_argument("--mesh-heat-blend-cut", type=float, default=0.12, help="Blend cutoff in [0,1].")
    rend.add_argument("--mesh-heat-blend-start", type=float, default=0.15, help="Start position within the colormap in [0,1].")

    heat = ap.add_argument_group("Heat")
    heat.add_argument("--heat-source", type=str, default="stl_infer", choices=["stl_infer", "external_icp", "both"])
    heat.add_argument("--tasks", type=str, default="custom_scalar", help="Comma list: raw_seg,raw_cls_saliency,custom_scalar")

    heat.add_argument("--sample-n", type=int, default=0, help="Sample points per mesh part (0=auto).")
    heat.add_argument("--sample-seed", type=int, default=0)
    heat.add_argument("--map-to-mesh", type=str, default="knn_mean", choices=["nearest", "knn_mean"])
    heat.add_argument("--knn-k", type=int, default=8)
    heat.add_argument("--vis-quantiles", type=str, default="0.02,0.98", help="Visualization contrast quantiles: 'qlo,qhi' or 'none'.")
    heat.add_argument("--vis-gamma", type=float, default=0.5, help="Visualization gamma: output = input**gamma (1 disables).")
    heat.add_argument("--heat-smooth-iters", type=int, default=1, help="kNN smoothing iterations for vertex heat (0 disables).")
    heat.add_argument("--heat-smooth-k", type=int, default=12, help="kNN for smoothing vertex heat.")
    heat.add_argument("--heat-smooth-alpha", type=float, default=0.6, help="Smoothing blend factor in [0,1].")

    heat.add_argument("--scalar", type=str, default="curvature_mean", help="When task=custom_scalar: z|x|y|curvature_mean|curvature_gaussian")

    heat.add_argument("--raw-seg-run-dir", type=Path, default=None, help="When task=raw_seg: run_dir under runs/raw_seg/... (contains train_config.json + model_best.pt).")
    heat.add_argument("--raw-seg-heat-mode", type=str, default="maxprob", choices=["maxprob", "entropy", "class_prob"])
    heat.add_argument("--raw-seg-class-id", type=int, default=1, help="When raw_seg_heat_mode=class_prob.")
    heat.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    heat.add_argument("--raw-cls-run-dir", type=Path, default=None, help="When task=raw_cls_saliency: run_dir under runs/raw_cls*/... (contains config.json + model_best.pt).")
    heat.add_argument("--raw-cls-target-class", type=str, default="pred", help="pred or integer class id.")

    heat.add_argument("--external-npz", type=Path, default=None, help="When heat_source includes external_icp: npz containing points + heat.")
    heat.add_argument("--icp-target", type=str, default="upperlower", choices=["all", "upperlower", "upper", "lower"])
    heat.add_argument("--icp-iters", type=int, default=50)
    heat.add_argument("--icp-threshold", type=float, default=1e-5)

    outg = ap.add_argument_group("Output")
    outg.add_argument("--out-dir", type=Path, default=Path("outputs/internal_db_heatmap"), help="Base output directory.")
    outg.add_argument("--screenshots", type=Path, default=None, help="If set, export fixed-view PNGs into this folder (or base folder when multiple jobs).")
    outg.add_argument("--screenshot-width", type=int, default=2200)
    outg.add_argument("--screenshot-height", type=int, default=1650)
    outg.add_argument("--screenshot-scale", type=float, default=2.0)
    outg.add_argument("--screenshot-crop", action=argparse.BooleanOptionalAction, default=True)
    outg.add_argument("--screenshot-crop-pad", type=int, default=40)
    outg.add_argument("--screenshot-crop-threshold", type=int, default=10)
    outg.add_argument("--dry-run", action="store_true", help="Resolve paths and print selection, but do not render.")

    args = ap.parse_args()

    theme = str(args.theme).strip().lower()
    if theme == "auto":
        theme = "dark" if args.screenshots is not None else "light"
    if theme not in {"light", "dark"}:
        raise SystemExit(f"Invalid theme: {args.theme!r}")
    if theme == "dark" and str(args.background).strip().lower() in {"#f7f7f7", "f7f7f7"}:
        args.background = "#0b1020"

    index_path = args.index
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path} (run scripts/internal_db/index_internal_db.py first)")
    records = _read_jsonl(index_path)
    if not records:
        raise SystemExit(f"Empty index: {index_path}")

    record = _pick_case(
        records,
        case_uid=(args.case_uid.strip() or None),
        query=(args.query.strip() or None),
        pick_latest=bool(args.pick_latest),
    )
    root = args.root.expanduser().resolve()
    case_uid = str(record.get("case_uid", "unknown"))

    show = {s.strip().lower() for s in str(args.show).split(",") if s.strip()}
    upper_path = _resolve_best_path(record, root=root, key="upper_stl")
    lower_path = _resolve_best_path(record, root=root, key="lower_stl")
    bite_path = _resolve_best_path(record, root=root, key="bite_stl")

    if args.dry_run:
        print(json.dumps({"case_uid": case_uid, "source_relpath": record.get("source_relpath")}, ensure_ascii=False))
        print("upper:", upper_path)
        print("lower:", lower_path)
        print("bite :", bite_path)
        return

    def _load_mesh(kind: str, path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        if not path.exists():
            raise SystemExit(f"Missing file for {kind}: {path}")
        poly = load_stl_as_polydata(path)
        prep = prepare_polydata(poly, align="none", decimate=float(args.decimate), smooth_iters=int(args.smooth_iters))
        return {"kind": kind, "mesh": prep.mesh}

    meshes: list[dict[str, Any]] = []
    if "upper" in show:
        m = _load_mesh("upper", upper_path)
        if m:
            meshes.append(m)
    if "lower" in show:
        m = _load_mesh("lower", lower_path)
        if m:
            meshes.append(m)
    if "bite" in show:
        m = _load_mesh("bite", bite_path)
        if m:
            meshes.append(m)
    if not meshes:
        raise SystemExit("Nothing to render (check --show and available STL assets).")

    present = {str(m["kind"]): m for m in meshes}

    def _subset(mode: str) -> list[dict[str, Any]]:
        mm = str(mode).strip().lower()
        if mm == "all":
            return meshes
        if mm == "upper":
            return [present["upper"]] if "upper" in present else meshes
        if mm == "lower":
            return [present["lower"]] if "lower" in present else meshes
        if mm == "upperlower":
            picked = [present[k] for k in ("upper", "lower") if k in present]
            return picked if picked else meshes
        raise SystemExit(f"Invalid subset mode: {mode!r}")

    R, t = compute_case_alignment([m["mesh"] for m in _subset(str(args.align_from))], mode=str(args.align))
    for m in meshes:
        apply_alignment(m["mesh"], R=R, t=t)

    # Bounds for camera framing.
    bounds: tuple[float, float, float, float, float, float] | None = None
    for m in _subset(str(args.camera_from)):
        b = m["mesh"].bounds
        if bounds is None:
            bounds = b
        else:
            bounds = (
                min(bounds[0], b[0]),
                max(bounds[1], b[1]),
                min(bounds[2], b[2]),
                max(bounds[3], b[3]),
                min(bounds[4], b[4]),
                max(bounds[5], b[5]),
            )
    if bounds is None:
        raise SystemExit("Failed to compute bounds.")

    center, _radius = bounds_center_radius(bounds)

    # Theme colors.
    if theme == "dark":
        enamel = "#efe9da"
        enamel2 = "#e2d9c8"
        bg_hex = str(args.background)
    else:
        enamel = "#f2efe6"
        enamel2 = "#f0eee8"
        bg_hex = str(args.background)

    tasks = [s.strip().lower() for s in str(args.tasks).split(",") if s.strip()]
    if not tasks:
        tasks = ["custom_scalar"]

    heat_sources: list[str]
    hs = str(args.heat_source).strip().lower()
    if hs == "both":
        heat_sources = ["stl_infer", "external_icp"]
    else:
        heat_sources = [hs]

    # Optional external heat.
    ext_points: np.ndarray | None = None
    ext_heat01: np.ndarray | None = None
    if "external_icp" in heat_sources:
        if args.external_npz is None:
            raise SystemExit("--external-npz is required when --heat-source includes external_icp")
        ext_points, ext_heat01 = _load_external_npz(Path(args.external_npz))

    # Lazy load models.
    raw_seg_model = None
    raw_seg_spec = None
    raw_cls_model = None
    raw_cls_spec = None

    out_base = Path(args.out_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    def _write_job(
        *,
        job_name: str,
        vertex_heat_by_kind: dict[str, np.ndarray],
        point_fields: list[HeatField],
        title_suffix: str,
    ) -> None:
        import plotly.graph_objects as go

        render_mode = str(args.render).strip().lower()
        traces: list[Any] = []
        showscale_used = False
        font_color = "#e5e7eb" if theme == "dark" else "#111827"

        # Light position follows camera view (in data coordinates).
        c0, r0 = bounds_center_radius(bounds)
        cam_pos0, _f0, _u0 = infer_camera(c0, r0, view="front")
        lightpos = {"x": float(cam_pos0[0] - c0[0]), "y": float(cam_pos0[1] - c0[1]), "z": float(cam_pos0[2] - c0[2])}
        lighting = (
            dict(ambient=0.60, diffuse=0.95, specular=0.18, roughness=0.70, fresnel=0.05)
            if theme == "dark"
            else dict(ambient=0.50, diffuse=0.95, specular=0.15, roughness=0.75, fresnel=0.05)
        )

        for m in meshes:
            kind = str(m["kind"])
            mesh = m["mesh"]
            x, y, z, i, j, k = _poly_to_plotly_triangles(mesh, center=center)

            base_col = enamel if kind == "upper" else enamel2
            vheat = vertex_heat_by_kind.get(kind)
            if vheat is None or render_mode == "points":
                traces.append(
                    go.Mesh3d(
                        name=kind,
                        x=x,
                        y=y,
                        z=z,
                        i=i,
                        j=j,
                        k=k,
                        color=base_col,
                        opacity=1.0,
                        flatshading=False,
                        showscale=False,
                        lighting=lighting,
                        lightposition=lightpos,
                    )
                )
                continue

            vheat01 = np.asarray(vheat, dtype=np.float32).reshape(-1)
            if vheat01.shape[0] != np.asarray(mesh.points).shape[0]:
                raise ValueError(f"vertex heat length mismatch for {kind}: {vheat01.shape[0]} vs {mesh.points.shape[0]}")

            traces.append(
                go.Mesh3d(
                    name=f"{kind}_heat",
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    intensity=(vheat01.astype(float).tolist()),
                    intensitymode="vertex",
                    colorscale=(
                        _blended_colorscale(
                            base_hex=base_col,
                            cmap=str(args.cmap),
                            cut=float(args.mesh_heat_blend_cut),
                            cmap_start=float(args.mesh_heat_blend_start),
                        )
                        if bool(args.mesh_heat_blend)
                        else str(args.cmap)
                    ),
                    cmin=0.0,
                    cmax=1.0,
                    showscale=(not showscale_used),
                    colorbar=dict(
                        title=dict(text="heat", font=dict(size=20, color=font_color)),
                        tickfont=dict(size=18, color=font_color),
                        thickness=30,
                        outlinewidth=0,
                    ),
                    opacity=1.0,
                    flatshading=False,
                    lighting=lighting,
                    lightposition=lightpos,
                )
            )
            showscale_used = True

        if str(args.render).strip().lower() in {"points", "both"}:
            for pf in point_fields:
                pts = np.asarray(pf.points, dtype=np.float32) - center.astype(np.float32)
                h01 = np.asarray(pf.heat01, dtype=np.float32).reshape(-1)
                if pts.shape[0] != h01.shape[0]:
                    continue
                traces.append(
                    go.Scatter3d(
                        name=pf.name,
                        x=pts[:, 0].astype(float).tolist(),
                        y=pts[:, 1].astype(float).tolist(),
                        z=pts[:, 2].astype(float).tolist(),
                        mode="markers",
                        marker=dict(
                            size=1.6,
                            color=h01.astype(float).tolist(),
                            colorscale=str(args.cmap),
                            cmin=0.0,
                            cmax=1.0,
                            opacity=0.85,
                        ),
                        hoverinfo="skip",
                    )
                )

        fig = go.Figure(data=traces)
        eye, upv = _build_camera(bounds, view="front", eye_mult=float(args.plotly_eye_mult))
        fig.update_layout(
            title=f"Intraoral Heatmap Viewer — {case_uid} — {title_suffix}",
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor=bg_hex,
            font=dict(color=font_color, size=18),
            scene=dict(
                bgcolor=bg_hex,
                aspectmode="data",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(eye=eye, up=upv, center={"x": 0.0, "y": 0.0, "z": 0.0}),
            ),
            legend=dict(itemsizing="constant"),
        )

        html_path = out_base / f"{case_uid}_{job_name}.html"
        fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)

        ss_base = Path(args.screenshots) if args.screenshots is not None else (out_base / "screens")
        ss_dir = (ss_base / f"{case_uid}_{job_name}").resolve()
        ss_dir.mkdir(parents=True, exist_ok=True)

        # Export screenshots (plotly + kaleido).
        try:
            import kaleido  # noqa: F401
        except Exception:
            print(f"[warn] kaleido not installed; skip PNG screenshots for {job_name}.")
            print(f"Wrote HTML: {html_path}")
            return

        _ensure_kaleido_chrome()
        fig.update_layout(title=None, margin=dict(l=0, r=0, t=0, b=0))

        for view in ["front", "left", "right", "top"]:
            v_eye, v_up = _build_camera(bounds, view=view, eye_mult=float(args.plotly_eye_mult))
            # Update light position per view to keep shading consistent.
            v_cam_pos, _vf, _vu = infer_camera(c0, r0, view=view)
            v_lightpos = {"x": float(v_cam_pos[0] - c0[0]), "y": float(v_cam_pos[1] - c0[1]), "z": float(v_cam_pos[2] - c0[2])}
            fig.update_traces(lightposition=v_lightpos, selector=dict(type="mesh3d"))
            fig.update_layout(scene_camera=dict(eye=v_eye, up=v_up, center={"x": 0.0, "y": 0.0, "z": 0.0}))
            out_png = ss_dir / f"{case_uid}_{job_name}_{view}.png"
            fig.write_image(str(out_png), width=int(args.screenshot_width), height=int(args.screenshot_height), scale=float(args.screenshot_scale))
            if bool(args.screenshot_crop):
                _crop_png(
                    out_png,
                    bg_hex=bg_hex,
                    threshold=int(args.screenshot_crop_threshold),
                    pad=int(args.screenshot_crop_pad),
                )

        print(f"Wrote: {html_path}")
        print(f"Screens: {ss_dir}")

    # Precompute ICP alignment if needed.
    icp_matrix: np.ndarray | None = None
    icp_cost: float | None = None
    ext_aligned: np.ndarray | None = None
    if ext_points is not None and ext_heat01 is not None and "external_icp" in heat_sources:
        target_pts_all: list[np.ndarray] = []
        for m in _subset(str(args.icp_target)):
            pts_s, _ = sample_points_on_polydata(m["mesh"], n=20000, seed=int(args.sample_seed))
            target_pts_all.append(pts_s)
        target_pts = np.concatenate(target_pts_all, axis=0) if target_pts_all else np.asarray(center, dtype=np.float32).reshape(1, 3)
        res = icp_align(
            ext_points,
            target_pts,
            max_iterations=int(args.icp_iters),
            threshold=float(args.icp_threshold),
            use_pca_init=True,
        )
        icp_matrix = res.matrix_4x4
        icp_cost = float(res.cost)
        ext_aligned = res.aligned

    # Jobs.
    for src in heat_sources:
        for task in tasks:
            job = f"{src}_{task}"
            vertex_heat: dict[str, np.ndarray] = {}
            point_fields: list[HeatField] = []
            title_parts: list[str] = [src, task]

            if src == "stl_infer":
                if task == "custom_scalar":
                    for m in meshes:
                        kind = str(m["kind"])
                        if kind not in {"upper", "lower"}:
                            continue
                        v01 = compute_custom_scalar_on_vertices(m["mesh"], kind=str(args.scalar))
                        vertex_heat[kind] = v01
                    title_parts.append(str(args.scalar))
                elif task == "raw_seg":
                    if args.raw_seg_run_dir is None:
                        raise SystemExit("--raw-seg-run-dir is required for task=raw_seg")
                    if raw_seg_model is None:
                        raw_seg_model, raw_seg_spec = load_raw_seg_model(Path(args.raw_seg_run_dir), device=str(args.device))
                    n_auto = int(getattr(raw_seg_spec, "n_points", 0) or 0) if raw_seg_spec is not None else 0
                    n_samp = int(args.sample_n) if int(args.sample_n) > 0 else (n_auto if n_auto > 0 else 8192)
                    for m in meshes:
                        kind = str(m["kind"])
                        if kind not in {"upper", "lower"}:
                            continue
                        pts_s, _ = sample_points_on_polydata(m["mesh"], n=n_samp, seed=int(args.sample_seed))
                        probs = infer_raw_seg_probs(raw_seg_model, pts_s, device=str(args.device))
                        heat01 = heat_from_probs(probs, mode=str(args.raw_seg_heat_mode), class_id=int(args.raw_seg_class_id))
                        heat01 = np.clip(heat01, 0.0, 1.0).astype(np.float32, copy=False)
                        v01 = map_point_scalar_to_vertices(
                            vertices=np.asarray(m["mesh"].points, dtype=np.float32),
                            points=pts_s,
                            scalar=heat01,
                            mode=str(args.map_to_mesh),
                            k=int(args.knn_k),
                        )
                        v01 = np.clip(v01, 0.0, 1.0).astype(np.float32, copy=False)
                        vertex_heat[kind] = v01
                        point_fields.append(HeatField(name=f"{kind}_heat", points=pts_s, heat01=heat01))
                    title_parts.append(f"run={Path(args.raw_seg_run_dir).name}")
                    title_parts.append(f"mode={args.raw_seg_heat_mode}")
                elif task == "raw_cls_saliency":
                    if args.raw_cls_run_dir is None:
                        raise SystemExit("--raw-cls-run-dir is required for task=raw_cls_saliency")
                    if raw_cls_model is None:
                        raw_cls_model, raw_cls_spec = load_raw_cls_model(Path(args.raw_cls_run_dir), device=str(args.device))
                    n_auto = int(getattr(raw_cls_spec, "n_points", 0) or 0) if raw_cls_spec is not None else 0
                    n_samp = int(args.sample_n) if int(args.sample_n) > 0 else (n_auto if n_auto > 0 else 4096)
                    tgt = str(args.raw_cls_target_class).strip().lower()
                    target_cls: int | str
                    target_cls = "pred" if tgt == "pred" else int(tgt)
                    extra_dim = int(getattr(raw_cls_spec, "extra_dim", 0) or 0) if raw_cls_spec is not None else 0
                    target_used: int | None = None
                    for m in meshes:
                        kind = str(m["kind"])
                        if kind not in {"upper", "lower"}:
                            continue
                        pts_s, _ = sample_points_on_polydata(m["mesh"], n=n_samp, seed=int(args.sample_seed))
                        sal, tcls = raw_cls_point_saliency(
                            raw_cls_model,
                            pts_s,
                            device=str(args.device),
                            target_class=target_cls,
                            extra_dim=extra_dim,
                        )
                        target_used = int(tcls)
                        heat01 = normalize01(sal)
                        v01 = map_point_scalar_to_vertices(
                            vertices=np.asarray(m["mesh"].points, dtype=np.float32),
                            points=pts_s,
                            scalar=heat01,
                            mode=str(args.map_to_mesh),
                            k=int(args.knn_k),
                        )
                        vertex_heat[kind] = v01
                        point_fields.append(HeatField(name=f"{kind}_saliency", points=pts_s, heat01=heat01))
                    title_parts.append(f"run={Path(args.raw_cls_run_dir).name}")
                    title_parts.append(f"target={target_used if target_used is not None else args.raw_cls_target_class}")
                else:
                    raise SystemExit(f"Unknown task: {task!r}")

            elif src == "external_icp":
                if ext_aligned is None or ext_heat01 is None:
                    raise SystemExit("External ICP requested but no external heat loaded/aligned")
                title_parts.append(f"icp_cost={icp_cost:.4g}" if icp_cost is not None else "icp")
                # Map aligned external heat onto each mesh part.
                for m in meshes:
                    kind = str(m["kind"])
                    if kind not in {"upper", "lower"}:
                        continue
                    v01 = map_point_scalar_to_vertices(
                        vertices=np.asarray(m["mesh"].points, dtype=np.float32),
                        points=np.asarray(ext_aligned, dtype=np.float32),
                        scalar=np.asarray(ext_heat01, dtype=np.float32),
                        mode=str(args.map_to_mesh),
                        k=int(args.knn_k),
                    )
                    vertex_heat[kind] = v01
                point_fields.append(HeatField(name="external_heat", points=np.asarray(ext_aligned, dtype=np.float32), heat01=np.asarray(ext_heat01, dtype=np.float32)))

            else:
                raise SystemExit(f"Unknown heat source: {src!r}")

            # Visualization contrast scaling (shared across parts for this job).
            if int(args.heat_smooth_iters) > 0 and vertex_heat:
                for m in meshes:
                    kind = str(m["kind"])
                    if kind not in vertex_heat:
                        continue
                    vertex_heat[kind] = smooth_vertex_scalar_knn(
                        np.asarray(m["mesh"].points, dtype=np.float32),
                        vertex_heat[kind],
                        k=int(args.heat_smooth_k),
                        iters=int(args.heat_smooth_iters),
                        alpha=float(args.heat_smooth_alpha),
                    )

            q = _parse_quantiles(str(args.vis_quantiles))
            if q is not None:
                qlo, qhi = q
                all_vals: list[np.ndarray] = []
                for v in vertex_heat.values():
                    vv = np.asarray(v, dtype=np.float32).reshape(-1)
                    if vv.size:
                        all_vals.append(vv)
                for pf in point_fields:
                    hh = np.asarray(pf.heat01, dtype=np.float32).reshape(-1)
                    if hh.size:
                        all_vals.append(hh)
                if all_vals:
                    cat = np.concatenate(all_vals, axis=0)
                    lo = float(np.quantile(cat, qlo))
                    hi = float(np.quantile(cat, qhi))
                    gamma = float(args.vis_gamma)
                    for k in list(vertex_heat.keys()):
                        vertex_heat[k] = _vis_scale_array(vertex_heat[k], lo=lo, hi=hi, gamma=gamma)
                    point_fields = [HeatField(name=pf.name, points=pf.points, heat01=_vis_scale_array(pf.heat01, lo=lo, hi=hi, gamma=gamma)) for pf in point_fields]

            _write_job(
                job_name=job,
                vertex_heat_by_kind=vertex_heat,
                point_fields=point_fields,
                title_suffix=" | ".join(title_parts),
            )

    # Also keep a pure-geometry viewer for reference.
    headless = not bool(os.environ.get("DISPLAY"))
    has_osmesa = ctypes.util.find_library("OSMesa") is not None
    if headless and not has_osmesa:
        pass


if __name__ == "__main__":
    try:
        main()
    except OptionalDependencyError as e:
        raise SystemExit(str(e)) from e
