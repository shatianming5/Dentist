from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_queue(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Queue must be a JSON list, got {type(data).__name__}")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Queue[{i}] must be an object, got {type(item).__name__}")
        out.append(item)
    return out


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _result_path(results_dir: Path, job_id: str, stage: str) -> Path:
    return results_dir / f"{job_id}-{stage}.json"


def _log_path(logs_dir: Path, job_id: str, stage: str) -> Path:
    return logs_dir / f"{job_id}-{stage}.log"


def main() -> None:
    ap = argparse.ArgumentParser(description="Show progress for a .rd_queue queue JSON.")
    ap.add_argument("--queue", type=str, default=".rd_queue/queue.json", help="Path to queue JSON.")
    ap.add_argument("--results-dir", type=str, default=".rd_queue/results", help="Results directory.")
    ap.add_argument("--logs-dir", type=str, default=".rd_queue/logs", help="Logs directory.")
    ap.add_argument("--max-name", type=int, default=88, help="Max characters to print for job name.")
    args = ap.parse_args()

    queue_path = Path(args.queue)
    results_dir = Path(args.results_dir)
    logs_dir = Path(args.logs_dir)

    jobs = _load_queue(queue_path)

    rows: list[tuple[bool, str, str, str]] = []
    completed = 0
    for job in jobs:
        job_id = _safe_str(job.get("id"))
        stage = _safe_str(job.get("stage"))
        name = _safe_str(job.get("name"))
        if not job_id or not stage:
            continue
        done = _result_path(results_dir, job_id, stage).exists()
        if done:
            completed += 1
        rows.append((done, job_id, stage, name))

    total = len(rows)
    print(f"queue={queue_path}  completed={completed}/{total}")

    next_job: tuple[bool, str, str, str] | None = None
    for row in rows:
        if not row[0]:
            next_job = row
            break

    if next_job is not None:
        _, job_id, stage, name = next_job
        print(f"next={job_id} stage={stage}")
        print(f"log={_log_path(logs_dir, job_id, stage)}")
        if name:
            print(f"name={name[: int(args.max_name)]}")

    for done, job_id, stage, name in rows:
        mark = "x" if done else " "
        if name:
            name = name.replace("\n", " ")
            name = name[: int(args.max_name)]
            print(f"[{mark}] {job_id} ({stage}) {name}")
        else:
            print(f"[{mark}] {job_id} ({stage})")


if __name__ == "__main__":
    main()

