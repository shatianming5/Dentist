#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import ctypes.util
import shutil
from pathlib import Path
from typing import Any

from internal_db.mesh_io import (
    OptionalDependencyError,
    apply_alignment,
    bounds_center_radius,
    compute_case_alignment,
    infer_camera,
    load_stl_as_polydata,
    prepare_polydata,
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
        # fallback: search in asset paths
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a single intraoral STL case to an interactive HTML viewer.")
    ap.add_argument("--index", type=Path, default=Path("metadata/internal_db/index.jsonl"), help="Index JSONL file.")
    ap.add_argument("--root", type=Path, default=Path("downloads/体内数据库/体内数据库"), help="Root folder (for resolving relative paths).")

    sel = ap.add_argument_group("Selection")
    sel.add_argument("--case-uid", type=str, default="", help="Select by case_uid.")
    sel.add_argument("--query", type=str, default="", help="Substring match on source_relpath / asset paths.")
    sel.add_argument("--pick-latest", action="store_true", help="Pick record with max mtime_max.")

    rend = ap.add_argument_group("Rendering")
    rend.add_argument("--show", type=str, default="upper,lower", help="Comma list: upper,lower,bite")
    rend.add_argument("--opacity-upper", type=float, default=1.0)
    rend.add_argument("--opacity-lower", type=float, default=1.0)
    rend.add_argument("--opacity-bite", type=float, default=0.35)
    rend.add_argument(
        "--bite-mode",
        type=str,
        default="points",
        choices=["mesh", "points", "off"],
        help="How to render bite scan. 'points' avoids z-fighting and is clearer in screenshots.",
    )
    rend.add_argument("--bite-max-points", type=int, default=60_000, help="Max points when --bite-mode=points (0 disables downsampling).")
    rend.add_argument("--bite-point-size", type=float, default=1.8, help="Marker size (px) when --bite-mode=points.")
    rend.add_argument("--theme", type=str, default="auto", choices=["auto", "light", "dark"], help="Viewer theme (colors/background).")
    rend.add_argument("--align", type=str, default="pca", choices=["none", "center", "pca"])
    rend.add_argument(
        "--align-from",
        type=str,
        default="upperlower",
        choices=["all", "upperlower", "upper", "lower"],
        help="Which parts to use when computing the shared alignment transform (default: upper/lower if present).",
    )
    rend.add_argument(
        "--camera-from",
        type=str,
        default="upperlower",
        choices=["all", "upperlower", "upper", "lower"],
        help="Which parts to use for camera framing (default: upper/lower if present).",
    )
    rend.add_argument("--decimate", type=float, default=0.3, help="Target reduction ratio in [0,1).")
    rend.add_argument("--smooth-iters", type=int, default=5)
    rend.add_argument("--background", type=str, default="#f7f7f7", help="Hex color, e.g. '#111827'.")
    rend.add_argument(
        "--plotly-eye-mult",
        type=float,
        default=0.6,
        help="Plotly camera distance multiplier (smaller -> closer/zoom in). Ignored for pyvista backend.",
    )
    rend.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "plotly", "pyvista"],
        help="HTML backend. 'pyvista' looks best but may require a working OpenGL context; 'plotly' works headless.",
    )

    outg = ap.add_argument_group("Output")
    outg.add_argument("--out", type=Path, default=None, help="Output .html path (default: outputs/internal_db_viewer/<case_uid>.html).")
    outg.add_argument("--screenshots", type=Path, default=None, help="If set, export fixed-view PNGs into this folder.")
    outg.add_argument("--screenshot-width", type=int, default=2200, help="Screenshot width in pixels (Plotly backend, before --screenshot-scale).")
    outg.add_argument("--screenshot-height", type=int, default=1650, help="Screenshot height in pixels (Plotly backend, before --screenshot-scale).")
    outg.add_argument("--screenshot-scale", type=float, default=2.0, help="Screenshot scale factor (Plotly backend).")
    outg.add_argument(
        "--screenshot-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop exported PNGs to remove large background margins (Plotly backend).",
    )
    outg.add_argument("--screenshot-crop-pad", type=int, default=40, help="Padding (px) when cropping screenshots.")
    outg.add_argument("--screenshot-crop-threshold", type=int, default=10, help="RGB delta threshold for detecting non-background.")
    outg.add_argument("--dry-run", action="store_true", help="Resolve paths and print selection, but do not render.")

    args = ap.parse_args()

    theme = str(args.theme).strip().lower()
    if theme == "auto":
        theme = "dark" if args.screenshots is not None else "light"
    if theme not in {"light", "dark"}:
        raise SystemExit(f"Invalid --theme: {args.theme!r}")
    # Apply theme defaults unless user explicitly changed the default background.
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

    out_path = args.out
    if out_path is None:
        out_path = Path("outputs/internal_db_viewer") / f"{case_uid}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _poly_to_plotly_triangles(
        poly: Any,
        *,
        center: Any | None = None,
    ) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int]]:
        pts = poly.points
        if center is not None:
            pts = pts - center
        faces = poly.faces
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

    def _poly_to_plotly_points(
        poly: Any,
        *,
        center: Any | None = None,
        max_points: int = 60_000,
    ) -> tuple[list[float], list[float], list[float]]:
        import numpy as np

        pts = poly.points
        if center is not None:
            pts = pts - center
        mp = int(max_points)
        if mp > 0 and pts.shape[0] > mp:
            rng = np.random.default_rng(0)
            idx = rng.choice(int(pts.shape[0]), size=mp, replace=False)
            pts = pts[idx]
        x = pts[:, 0].astype(float).tolist()
        y = pts[:, 1].astype(float).tolist()
        z = pts[:, 2].astype(float).tolist()
        return x, y, z

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

    def _crop_png(path: Path) -> None:
        from PIL import Image
        import numpy as np

        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im)

        bg = _hex_to_rgb(str(args.background)) or tuple(int(x) for x in arr[0, 0, :3])
        bg_arr = np.asarray(bg, dtype=np.int16)
        diff = np.abs(arr.astype(np.int16) - bg_arr[None, None, :])
        thr = int(args.screenshot_crop_threshold)
        mask = (diff.max(axis=2) > thr)
        if not bool(mask.any()):
            return
        ys, xs = np.where(mask)
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1

        pad = int(args.screenshot_crop_pad)
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(arr.shape[0], y1 + pad)
        x1 = min(arr.shape[1], x1 + pad)

        with Image.open(path) as im2:
            im2 = im2.convert("RGB")
            cropped = im2.crop((x0, y0, x1, y1))
            cropped.save(path)

    def _collect_prepped() -> tuple[list[dict[str, Any]], tuple[float, float, float, float, float, float]]:
        meshes: list[dict[str, Any]] = []

        def _add_mesh(kind: str, path: Path | None, *, color: str, opacity: float, render_as: str = "mesh") -> None:
            if path is None:
                return
            if not path.exists():
                raise SystemExit(f"Missing file for {kind}: {path}")
            poly = load_stl_as_polydata(path)
            prep = prepare_polydata(poly, align="none", decimate=float(args.decimate), smooth_iters=int(args.smooth_iters))
            meshes.append(
                {"kind": kind, "mesh": prep.mesh, "color": color, "opacity": float(opacity), "render_as": str(render_as)}
            )

        # Enamel-like palette (warm ivory), tuned per theme for contrast.
        if theme == "dark":
            enamel = "#efe9da"
            enamel2 = "#e2d9c8"
            bite_col = "#60a5fa"
        else:
            enamel = "#f2efe6"
            enamel2 = "#f0eee8"
            bite_col = "#b9d6ff"

        if "upper" in show:
            _add_mesh("upper", upper_path, color=enamel, opacity=args.opacity_upper)
        if "lower" in show:
            _add_mesh("lower", lower_path, color=enamel2, opacity=args.opacity_lower)
        if "bite" in show:
            bite_mode = str(args.bite_mode).strip().lower()
            if bite_mode == "off":
                pass
            elif bite_mode in {"mesh", "points"}:
                _add_mesh("bite", bite_path, color=bite_col, opacity=args.opacity_bite, render_as=bite_mode)
            else:
                raise SystemExit(f"Invalid --bite-mode: {args.bite_mode!r}")

        if not meshes:
            raise SystemExit("Nothing to render (check --show and available STL assets).")

        def _pick_subset(mode: str) -> list[dict[str, Any]]:
            m = str(mode).strip().lower()
            present = {str(mm["kind"]): mm for mm in meshes}
            if m == "all":
                return meshes
            if m == "upper":
                return [present["upper"]] if "upper" in present else meshes
            if m == "lower":
                return [present["lower"]] if "lower" in present else meshes
            if m == "upperlower":
                picked = [present[k] for k in ("upper", "lower") if k in present]
                return picked if picked else meshes
            raise SystemExit(f"Invalid subset mode: {mode!r}")

        # Apply ONE consistent alignment transform to all parts to preserve relative geometry.
        align_subset = _pick_subset(str(args.align_from))
        R, t = compute_case_alignment([m["mesh"] for m in align_subset], mode=str(args.align))
        for m in meshes:
            apply_alignment(m["mesh"], R=R, t=t)

        camera_bounds: tuple[float, float, float, float, float, float] | None = None
        for m in _pick_subset(str(args.camera_from)):
            b = m["mesh"].bounds
            if camera_bounds is None:
                camera_bounds = b
            else:
                camera_bounds = (
                    min(camera_bounds[0], b[0]),
                    max(camera_bounds[1], b[1]),
                    min(camera_bounds[2], b[2]),
                    max(camera_bounds[3], b[3]),
                    min(camera_bounds[4], b[4]),
                    max(camera_bounds[5], b[5]),
                )
        if camera_bounds is None:
            raise SystemExit("Failed to compute bounds after alignment.")
        return meshes, camera_bounds

    def _export_plotly(meshes: list[dict[str, Any]], bounds: tuple[float, float, float, float, float, float]) -> None:
        import plotly.graph_objects as go

        center, radius = bounds_center_radius(bounds)
        cam_pos, focal, up = infer_camera(center, radius, view="front")
        rad = float(radius) if float(radius) > 1e-6 else 1.0
        eye_mult = float(args.plotly_eye_mult)
        eye_vec = (
            (float(cam_pos[0] - focal[0])) / rad * eye_mult,
            (float(cam_pos[1] - focal[1])) / rad * eye_mult,
            (float(cam_pos[2] - focal[2])) / rad * eye_mult,
        )
        eye = {"x": eye_vec[0], "y": eye_vec[1], "z": eye_vec[2]}
        lightpos = {"x": float(cam_pos[0] - center[0]), "y": float(cam_pos[1] - center[1]), "z": float(cam_pos[2] - center[2])}

        lighting = (
            dict(ambient=0.28, diffuse=0.95, specular=0.25, roughness=0.6, fresnel=0.05)
            if theme == "dark"
            else dict(ambient=0.38, diffuse=0.95, specular=0.22, roughness=0.65, fresnel=0.05)
        )

        traces = []
        for m in meshes:
            render_as = str(m.get("render_as", "mesh")).strip().lower()
            if render_as == "points":
                x, y, z = _poly_to_plotly_points(m["mesh"], center=center, max_points=int(args.bite_max_points))
                traces.append(
                    go.Scatter3d(
                        name=str(m["kind"]),
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=dict(
                            size=float(args.bite_point_size),
                            color=str(m["color"]),
                            opacity=float(m["opacity"]),
                        ),
                        hoverinfo="skip",
                    )
                )
            else:
                x, y, z, i, j, k = _poly_to_plotly_triangles(m["mesh"], center=center)
                traces.append(
                    go.Mesh3d(
                        name=str(m["kind"]),
                        x=x,
                        y=y,
                        z=z,
                        i=i,
                        j=j,
                        k=k,
                        color=str(m["color"]),
                        opacity=float(m["opacity"]),
                        flatshading=False,
                        showscale=False,
                        lighting=lighting,
                        lightposition=lightpos,
                    )
                )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"Intraoral STL Viewer — {case_uid}",
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor=str(args.background),
            scene=dict(
                bgcolor=str(args.background),
                aspectmode="data",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(
                    eye=eye,
                    up={"x": float(up[0]), "y": float(up[1]), "z": float(up[2])},
                    center={"x": 0.0, "y": 0.0, "z": 0.0},
                ),
            ),
            legend=dict(itemsizing="constant"),
        )
        fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)

        if args.screenshots is not None:
            # Remove title/margins for exported PNGs so the tooth fills the frame.
            fig.update_layout(title=None, margin=dict(l=0, r=0, t=0, b=0))

            ss_dir = Path(args.screenshots)
            ss_dir.mkdir(parents=True, exist_ok=True)
            # Requires kaleido. If missing, keep HTML and print a helpful note.
            try:
                import kaleido  # noqa: F401
            except Exception:
                print("Note: kaleido not installed; skip PNG screenshots. Install: pip install kaleido")
            else:
                # Help kaleido/choreographer find Chrome in common local install paths.
                if not os.environ.get("BROWSER_PATH"):
                    chrome_candidates = [
                        Path.home() / ".local/share/plotly/chrome-linux64/chrome",
                        Path.home() / ".local/share/kaleido/chrome-linux64/chrome",
                    ]
                    for c in chrome_candidates:
                        if c.exists() and os.access(c, os.X_OK):
                            os.environ["BROWSER_PATH"] = str(c)
                            break
                    if not os.environ.get("BROWSER_PATH"):
                        # Last resort: try system chrome
                        for exe in ["google-chrome", "google-chrome-stable", "chrome", "chromium", "chromium-browser"]:
                            p = shutil.which(exe)
                            if p:
                                os.environ["BROWSER_PATH"] = p
                                break

                for view in ["front", "left", "right", "top"]:
                    v_cam_pos, v_focal, v_up = infer_camera(center, radius, view=view)
                    v_eye_vec = (
                        float(v_cam_pos[0] - v_focal[0]) / rad * eye_mult,
                        float(v_cam_pos[1] - v_focal[1]) / rad * eye_mult,
                        float(v_cam_pos[2] - v_focal[2]) / rad * eye_mult,
                    )
                    v_eye = {"x": v_eye_vec[0], "y": v_eye_vec[1], "z": v_eye_vec[2]}
                    v_lightpos = {"x": float(v_cam_pos[0] - center[0]), "y": float(v_cam_pos[1] - center[1]), "z": float(v_cam_pos[2] - center[2])}
                    fig.update_layout(
                        scene_camera=dict(
                            eye=v_eye,
                            up={"x": float(v_up[0]), "y": float(v_up[1]), "z": float(v_up[2])},
                            center={"x": 0.0, "y": 0.0, "z": 0.0},
                        )
                    )
                    fig.update_traces(lightposition=v_lightpos, selector=dict(type="mesh3d"))
                    fig.write_image(
                        str(ss_dir / f"{case_uid}_{view}.png"),
                        width=int(args.screenshot_width),
                        height=int(args.screenshot_height),
                        scale=float(args.screenshot_scale),
                    )
                    if bool(args.screenshot_crop):
                        _crop_png(ss_dir / f"{case_uid}_{view}.png")

        print(f"Wrote: {out_path} (backend=plotly)")

    def _export_pyvista(meshes: list[dict[str, Any]], bounds: tuple[float, float, float, float, float, float]) -> None:
        try:
            pv = __import__("pyvista")
        except Exception:
            raise SystemExit("Missing pyvista/vtk. Install: pip install -r configs/env/requirements_vis.txt")

        plotter = pv.Plotter(off_screen=True, window_size=(1400, 1000))
        plotter.set_background(str(args.background))
        try:
            plotter.enable_lightkit()
        except Exception:
            pass

        for m in meshes:
            plotter.add_mesh(
                m["mesh"],
                name=str(m["kind"]),
                color=str(m["color"]),
                opacity=float(m["opacity"]),
                smooth_shading=True,
                specular=0.35,
                specular_power=25.0,
            )

        center, radius = bounds_center_radius(bounds)
        plotter.camera_position = infer_camera(center, radius, view="front")
        try:
            plotter.reset_camera()
        except Exception:
            pass

        if args.screenshots is not None:
            ss_dir = Path(args.screenshots)
            ss_dir.mkdir(parents=True, exist_ok=True)
            for view in ["front", "left", "right", "top"]:
                plotter.camera_position = infer_camera(center, radius, view=view)
                plotter.render()
                plotter.screenshot(str(ss_dir / f"{case_uid}_{view}.png"))

        try:
            plotter.export_html(str(out_path))
        finally:
            try:
                plotter.close()
            except Exception:
                pass

        print(f"Wrote: {out_path} (backend=pyvista)")

    meshes, bounds = _collect_prepped()

    backend = str(args.backend).strip().lower()
    if backend == "plotly":
        _export_plotly(meshes, bounds)
        return
    if backend == "pyvista":
        _export_pyvista(meshes, bounds)
        return
    if backend == "auto":
        headless = not bool(os.environ.get("DISPLAY"))
        has_osmesa = ctypes.util.find_library("OSMesa") is not None
        if headless and not has_osmesa:
            _export_plotly(meshes, bounds)
            return
        try:
            _export_pyvista(meshes, bounds)
            return
        except Exception as e:
            print(f"PyVista backend failed ({type(e).__name__}: {e}). Falling back to Plotly...")
            _export_plotly(meshes, bounds)
            return
    raise SystemExit(f"Unknown backend: {args.backend!r}")


if __name__ == "__main__":
    try:
        main()
    except OptionalDependencyError as e:
        raise SystemExit(str(e)) from e
