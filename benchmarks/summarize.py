#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCENARIO_LABELS = {
    "concurrency": "High concurrency (1024 input / 256 output)",
    "long_context": "Long context (128 output, concurrency 4)",
}
PARALLELISM_LABELS = {"tp4": "TP4", "pp4": "PP4"}
PRECISION_LABELS = {"bf16": "BF16 baseline", "w8a8": "W8A8"}
PRECISION_LABELS["w4a8"] = "W4A8 dynamic (random weights)"
PRECISION_ORDER = ("bf16", "w8a8", "w4a8")
COLORS = {"bf16": "#2463a6", "w8a8": "#d24b40", "w4a8": "#16836a"}
MARKERS = {"bf16": "o", "w8a8": "s", "w4a8": "^"}


def load_records(run_dir: Path):
    records = []
    for path in sorted(run_dir.glob("*_*/client/*.json")):
        if path.name.endswith(".pytorch.json"):
            continue
        data = json.loads(path.read_text())
        data["_path"] = str(path.relative_to(run_dir))
        for field in ("x_value", "input_tokens", "output_tokens", "repeat"):
            data[field] = int(data[field])
        records.append(data)
    return records


def aggregate(records):
    grouped = {}
    for record in records:
        key = (
            record["parallelism"],
            record["scenario"],
            record["precision"],
            record["x_value"],
        )
        grouped.setdefault(key, []).append(record)

    rows = []
    metrics = (
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "median_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "p90_tpot_ms",
        "p99_tpot_ms",
        "median_e2el_ms",
        "p90_e2el_ms",
        "p99_e2el_ms",
    )
    for key, values in sorted(grouped.items()):
        first = values[0]
        row = {
            "parallelism": key[0],
            "scenario": key[1],
            "precision": key[2],
            "x_value": key[3],
            "input_tokens": first["input_tokens"],
            "output_tokens": first["output_tokens"],
            "max_concurrency": first["max_concurrency"],
            "repeats": len(values),
            "completed": sum(int(item.get("completed", 0)) for item in values),
            "failed": sum(int(item.get("failed", 0)) for item in values),
            "hf_overrides": first.get("hf_overrides", "none"),
        }
        for metric in metrics:
            samples = [
                float(item[metric])
                for item in values
                if item.get(metric) is not None
            ]
            row[metric] = statistics.median(samples) if samples else ""
            row[f"{metric}_min"] = min(samples) if samples else ""
            row[f"{metric}_max"] = max(samples) if samples else ""
        rows.append(row)
    return rows


def load_memory_rows(source_dirs):
    rows = []
    for run_dir in source_dirs:
        selected_npus = set()
        run_env = run_dir / "run.env"
        if run_env.exists():
            for line in run_env.read_text().splitlines():
                if line.startswith("selected_npu_ids="):
                    selected_npus = {
                        int(value)
                        for value in line.split("=", 1)[1].split()
                    }
        for config_dir in sorted(run_dir.glob("*_*_*")):
            if not config_dir.is_dir():
                continue
            parts = config_dir.name.split("_", 2)
            if len(parts) != 3:
                continue
            parallelism, precision, scenario = parts
            if (
                parallelism not in PARALLELISM_LABELS
                or precision not in PRECISION_LABELS
                or scenario not in SCENARIO_LABELS
            ):
                continue
            server_log = config_dir / "server.log"
            if not server_log.exists():
                continue
            log_text = server_log.read_text(errors="replace")
            weights = [
                float(value)
                for value in re.findall(
                    r"Loading model weights took ([0-9.]+) GB", log_text
                )
            ]
            available = [
                int(value)
                for value in re.findall(
                    r"Available memory: ([0-9]+), total memory:", log_text
                )
            ]
            kv_tokens = [
                int(value.replace(",", ""))
                for value in re.findall(
                    r"GPU KV cache size: ([0-9,]+) tokens", log_text
                )
            ]

            peak_hbm_mb = ""
            sample_path = config_dir / "npu-samples.csv"
            if sample_path.exists():
                with sample_path.open(newline="", encoding="utf-8") as handle:
                    samples = [
                        int(row["hbm_used_mb"])
                        for row in csv.DictReader(handle)
                        if not selected_npus
                        or int(row["npu_id"]) in selected_npus
                    ]
                if samples:
                    peak_hbm_mb = max(samples)

            rows.append(
                {
                    "parallelism": parallelism,
                    "scenario": scenario,
                    "precision": precision,
                    "model_weights_gb_per_rank": (
                        statistics.mean(weights) if weights else ""
                    ),
                    "available_memory_gib_per_rank": (
                        statistics.mean(available) / 1024**3
                        if available
                        else ""
                    ),
                    "kv_cache_tokens": max(kv_tokens) if kv_tokens else "",
                    "peak_hbm_gib_per_card": (
                        peak_hbm_mb / 1024 if peak_hbm_mb != "" else ""
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["parallelism"],
            row["scenario"],
            PRECISION_ORDER.index(row["precision"]),
        ),
    )


def write_csv(rows, path: Path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def point_map(rows):
    return {
        (
            row["parallelism"],
            row["scenario"],
            row["precision"],
            row["x_value"],
        ): row
        for row in rows
    }


def plot_chart(rows, run_dir: Path, parallelism: str, scenario: str):
    selected = [
        row
        for row in rows
        if row["parallelism"] == parallelism and row["scenario"] == scenario
    ]
    if not selected:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for precision in PRECISION_ORDER:
        points = sorted(
            (row for row in selected if row["precision"] == precision),
            key=lambda row: row["x_value"],
        )
        if not points:
            continue
        x = [row["x_value"] for row in points]
        for axis, metric in zip(
            axes, ("output_throughput", "total_token_throughput")
        ):
            axis.plot(
                x,
                [row[metric] for row in points],
                label=PRECISION_LABELS[precision],
                color=COLORS[precision],
                marker=MARKERS[precision],
                linewidth=2,
                markersize=6,
            )

    if scenario == "concurrency":
        x_label = "Closed-loop concurrency"
        ticks = [1, 4, 8, 16, 32, 64]
    else:
        x_label = "Total context length (tokens)"
        ticks = [4096, 8192, 16384, 32768, 40960, 65536]
        for axis in axes:
            axis.axvline(
                40960,
                color="#666666",
                linestyle="--",
                linewidth=1,
                label="Native 40K limit",
            )
            axis.axvspan(40960, 65536, color="#777777", alpha=0.06)
            axis.text(
                64500,
                0.97,
                "64K uses static YaRN",
                transform=axis.get_xaxis_transform(),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=8,
                color="#555555",
            )

    for axis, title in zip(
        axes, ("Output throughput", "Total token throughput")
    ):
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel("tokens/s")
        axis.set_xticks(ticks)
        axis.grid(True, alpha=0.25)
        axis.legend()
    if scenario == "long_context":
        axes[0].set_xticklabels(["4K", "8K", "16K", "32K", "40K", "64K"])
        axes[1].set_xticklabels(["4K", "8K", "16K", "32K", "40K", "64K"])

    fig.suptitle(
        f"QwQ-32B {PARALLELISM_LABELS[parallelism]} - "
        f"{SCENARIO_LABELS[scenario]}"
    )
    fig.tight_layout()
    output = run_dir / f"{parallelism}_{scenario}_throughput.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt(value):
    return "N/A" if value == "" else f"{float(value):.2f}"


def write_report(rows, memory_rows, run_dir: Path, source_dirs):
    points = point_map(rows)
    lines = [
        "# QwQ-32B Four-NPU Throughput",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "Source directories: "
        + ", ".join(f"`{path}`" for path in source_dirs)
        + ".",
        "",
        "BF16 is the baseline. Each point has one timed repeat, so these are "
        "machine-validation results rather than variance-qualified medians. "
        "Output throughput and total token throughput are both reported because "
        "long-context prefill can dominate total token rate.",
        "",
        "ACL Graph is enabled and prefix caching is disabled. QwQ-32B declares "
        "a native 40K context. Points through 40K use the native model config; "
        "64K (*) uses the model README's static YaRN factor 4.0 with "
        "`max_position_embeddings=131072`. It is a throughput measurement, "
        "not a 64K model-quality claim.",
        "",
    ]
    if memory_rows:
        lines.extend(
            [
                "## Memory",
                "",
                "| Parallel | Scenario | Precision | Weights/rank GB | "
                "KV budget/rank GiB | KV capacity tokens | Peak HBM/card GiB |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in memory_rows:
            lines.append(
                f"| {row['parallelism'].upper()} | {row['scenario']} | "
                f"{row['precision'].upper()} | "
                f"{fmt(row['model_weights_gb_per_rank'])} | "
                f"{fmt(row['available_memory_gib_per_rank'])} | "
                f"{row['kv_cache_tokens'] or 'N/A'} | "
                f"{fmt(row['peak_hbm_gib_per_card'])} |"
            )
        lines.extend(
            [
                "",
                "The server uses `gpu-memory-utilization=0.90`; memory saved by "
                "quantized weights is normally reassigned to KV cache. Peak HBM "
                "therefore measures the configured pool, while weights/rank and "
                "KV capacity expose the useful memory difference.",
                "",
            ]
        )
    for parallelism in ("tp4", "pp4"):
        for scenario in ("concurrency", "long_context"):
            subset = sorted(
                (
                    row
                    for row in rows
                    if row["parallelism"] == parallelism
                    and row["scenario"] == scenario
                ),
                key=lambda row: (
                    row["x_value"],
                    PRECISION_ORDER.index(row["precision"]),
                ),
            )
            if not subset:
                continue
            lines.extend(
                [
                    f"## {PARALLELISM_LABELS[parallelism]} "
                    f"{SCENARIO_LABELS[scenario]}",
                    "",
                    "| Point | Precision | Output tok/s | Total tok/s | "
                    "TTFT p50 ms | TPOT p50 ms | Failed | vs BF16 output |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in subset:
                ratio = ""
                if row["precision"] != "bf16":
                    baseline = points.get(
                        (
                            parallelism,
                            scenario,
                            "bf16",
                            row["x_value"],
                        )
                    )
                    if baseline and baseline["output_throughput"]:
                        ratio = (
                            float(row["output_throughput"])
                            / float(baseline["output_throughput"])
                        )
                point = (
                    str(row["x_value"])
                    if scenario == "concurrency"
                    else (
                        "64K*"
                        if row["x_value"] == 65536
                        else f"{row['x_value'] // 1024}K"
                    )
                )
                lines.append(
                    f"| {point} | {row['precision'].upper()} | "
                    f"{fmt(row['output_throughput'])} | "
                    f"{fmt(row['total_token_throughput'])} | "
                    f"{fmt(row['median_ttft_ms'])} | "
                    f"{fmt(row['median_tpot_ms'])} | {row['failed']} | "
                    f"{fmt(ratio)} |"
                )
            lines.extend(
                [
                    "",
                    f"![{parallelism} {scenario}]"
                    f"({parallelism}_{scenario}_throughput.png)",
                    "",
                ]
            )

    startup_failures = sorted(run_dir.glob("*_*/status.txt"))
    failed_configs = [
        path.parent.name
        for path in startup_failures
        if path.read_text().strip() != "completed"
    ]
    lines.extend(["## Failures", ""])
    if failed_configs:
        lines.extend(f"- `{name}` did not complete." for name in failed_configs)
    else:
        lines.append("No configuration-level failures were recorded.")
    native_failure_logs = sorted(
        run_dir.glob("*_*/server.native64-failure.log")
    )
    if native_failure_logs:
        lines.extend(
            [
                "",
                "Diagnostic note: native 64K attempts failed with Ascend "
                "`GatherV3` position indices outside `[0, 40960)`. Those logs "
                "are preserved as:",
            ]
        )
        lines.extend(
            f"- `{path.relative_to(run_dir)}`" for path in native_failure_logs
        )
        rope_only_log = (
            run_dir
            / "tp4_bf16_long_context"
            / "server.yarn-rope-only-failure.log"
        )
        if rope_only_log.exists():
            lines.extend(
                [
                    "",
                    "Setting only `rope_scaling` did not enlarge the Ascend "
                    "precomputed table. The corrected 64K runs also override "
                    "`max_position_embeddings`; the intermediate diagnostic "
                    f"is `{rope_only_log.relative_to(run_dir)}`.",
                ]
            )
    lines.append("")
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--include-run-dir",
        action="append",
        default=[],
        type=Path,
        help="Include records from another run in the generated comparison.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source_dirs = [path.resolve() for path in args.include_run_dir] + [run_dir]
    records = []
    for source_dir in source_dirs:
        records.extend(load_records(source_dir))
    if not records:
        raise SystemExit(
            "No benchmark JSON files found under "
            + ", ".join(str(path) for path in source_dirs)
        )
    rows = aggregate(records)
    memory_rows = load_memory_rows(source_dirs)
    write_csv(rows, run_dir / "metrics.csv")
    write_csv(memory_rows, run_dir / "memory.csv")
    for parallelism in ("tp4", "pp4"):
        for scenario in ("concurrency", "long_context"):
            plot_chart(rows, run_dir, parallelism, scenario)
    write_report(rows, memory_rows, run_dir, source_dirs)
    print(f"Wrote summary for {len(records)} benchmark records to {run_dir}")


if __name__ == "__main__":
    main()
