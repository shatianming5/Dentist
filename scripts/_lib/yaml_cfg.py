from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: PyYAML. Install via `cd configs/env && pip install -r requirements.txt` "
        "or add PyYAML to your environment."
    ) from e


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (update or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def read_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise SystemExit(f"Config must be a YAML mapping, got {type(obj).__name__}: {path}")
    return obj


def load_with_defaults(path: Path, *, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in _stack:
        chain = " -> ".join(str(p) for p in _stack + (path,))
        raise SystemExit(f"Cycle detected in defaults chain: {chain}")

    cfg = read_yaml(path)
    defaults = cfg.get("defaults") or []
    if defaults and not isinstance(defaults, list):
        raise SystemExit(f"`defaults` must be a list: {path}")

    merged: dict[str, Any] = {}
    for item in defaults:
        if not isinstance(item, str) or not item.strip():
            raise SystemExit(f"Invalid defaults item {item!r} in {path}")
        inc = (path.parent / item).expanduser().resolve() if not Path(item).is_absolute() else Path(item).expanduser().resolve()
        merged = deep_merge(merged, load_with_defaults(inc, _stack=_stack + (path,)))

    cfg2 = dict(cfg)
    cfg2.pop("defaults", None)
    merged = deep_merge(merged, cfg2)
    return merged


_INTERP_RE = re.compile(r"\$\{([^}]+)\}")


def _get_by_path(cfg: dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def interpolate_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    def _walk(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, str):
            m = _INTERP_RE.fullmatch(x.strip())
            if m:
                key = m.group(1).strip()
                try:
                    return _get_by_path(cfg, key)
                except Exception:
                    return x

            def _repl(mm: re.Match[str]) -> str:
                key = mm.group(1).strip()
                try:
                    v = _get_by_path(cfg, key)
                except Exception:
                    return mm.group(0)
                return str(v)

            return _INTERP_RE.sub(_repl, x)
        return x

    return _walk(cfg)


def set_nested(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def write_yaml(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

