#!/usr/bin/env python3

import argparse
import csv
import statistics
from pathlib import Path


OP_TYPE = "WeightQuantBatchMatmulV2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create compact summaries from W4A8 CANN profiler CSVs."
    )
    parser.add_argument("--pergroup-dir", type=Path, required=True)
    parser.add_argument("--vllm-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize_pergroup(root):
    rows = []
    for m_dir in sorted(
        root.glob("m[0-9]*"), key=lambda path: int(path.name[1:])
    ):
        detail_paths = list(
            m_dir.glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv")
        )
        if len(detail_paths) != 1:
            raise RuntimeError(
                f"Expected one kernel_details.csv under {m_dir}, "
                f"found {len(detail_paths)}"
            )
        matches = [
            row
            for row in read_csv(detail_paths[0])
            if row.get("Type") == OP_TYPE
        ]
        durations = [float(row["Duration(us)"]) for row in matches]
        if not durations:
            raise RuntimeError(f"No {OP_TYPE} records under {m_dir}")
        sample = matches[0]
        rows.append(
            {
                "m": int(m_dir.name[1:]),
                "count": len(matches),
                "total_us": f"{sum(durations):.3f}",
                "average_us": f"{statistics.mean(durations):.3f}",
                "min_us": f"{min(durations):.3f}",
                "max_us": f"{max(durations):.3f}",
                "core_type": sample["Accelerator Core"],
                "input_data_types": sample["Input Data Types"],
                "input_formats": sample["Input Formats"],
            }
        )
    write_csv(root / "kernel_summary.csv", rows)


def summarize_vllm(root):
    rows = []
    for path in sorted(
        root.glob("trace/*/ASCEND_PROFILER_OUTPUT/op_statistic.csv")
    ):
        matches = [
            row for row in read_csv(path) if row.get("OP Type") == OP_TYPE
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {OP_TYPE} statistic in {path}, "
                f"found {len(matches)}"
            )
        row = matches[0]
        details = [
            item
            for item in read_csv(path.with_name("kernel_details.csv"))
            if item.get("Type") == OP_TYPE
        ]
        formats = sorted({item["Input Formats"] for item in details})
        data_types = sorted({item["Input Data Types"] for item in details})
        shapes = {item["Input Shapes"] for item in details}
        rows.append(
            {
                "device_id": row["Device_id"],
                "count": row["Count"],
                "total_us": row["Total Time(us)"],
                "average_us": row["Avg Time(us)"],
                "min_us": row["Min Time(us)"],
                "max_us": row["Max Time(us)"],
                "kernel_time_ratio_percent": row["Ratio(%)"],
                "unique_shapes": len(shapes),
                "input_data_types": "|".join(data_types),
                "input_formats": "|".join(formats),
            }
        )
    rows.sort(key=lambda row: int(row["device_id"]))
    write_csv(root / "weight_quant_summary.csv", rows)


def main():
    args = parse_args()
    summarize_pergroup(args.pergroup_dir.resolve())
    summarize_vllm(args.vllm_dir.resolve())


if __name__ == "__main__":
    main()
