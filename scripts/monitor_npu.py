#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import time


DEVICE_RE = re.compile(r"^\|\s*(\d+)\s+910B4")
USAGE_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*[^|]+\|\s*(\d+)\s+"
    r"\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)"
)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sample():
    output = subprocess.run(
        ["npu-smi", "info"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    device = None
    rows = []
    for line in output.splitlines():
        match = DEVICE_RE.match(line)
        if match:
            device = int(match.group(1))
            continue
        match = USAGE_RE.match(line)
        if device is not None and match:
            rows.append(
                {
                    "npu_id": device,
                    "aicore_percent": int(match.group(1)),
                    "hbm_used_mb": int(match.group(2)),
                    "hbm_total_mb": int(match.group(3)),
                }
            )
            device = None
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "timestamp_utc",
                "npu_id",
                "aicore_percent",
                "hbm_used_mb",
                "hbm_total_mb",
            ),
        )
        writer.writeheader()
        while process_alive(args.watch_pid):
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                for row in sample():
                    writer.writerow({"timestamp_utc": timestamp, **row})
                handle.flush()
            except (OSError, subprocess.SubprocessError):
                pass
            time.sleep(args.interval)


if __name__ == "__main__":
    main()

