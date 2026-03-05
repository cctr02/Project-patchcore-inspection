"""
aggregate_results.py — Aggregate all PatchCore experiment results into one CSV.

Usage:
    python contribution/aggregate_results.py
    python contribution/aggregate_results.py --results_root /path/to/results
    python contribution/aggregate_results.py --out summary.csv

Output format (one row per experiment):
    dataset | experiment | instance_auroc_Mean | instance_auroc_<subset1> | ...
                         | full_pixel_auroc_Mean | ...
                         | anomaly_pixel_auroc_Mean | ...

Within each dataset group, rows are sorted by instance_auroc_Mean descending.
Subset columns within each metric group: Mean first, then alphabetical.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple


METRICS = ["instance_auroc", "full_pixel_auroc", "anomaly_pixel_auroc"]

DATASET_GROUPS = {
    "MVTecAD_Results": "MVTecAD",
    "VisA_Results":    "VisA",
}


def find_results_csvs(results_root: Path) -> List[Tuple[str, str, Path]]:
    """
    Returns list of (dataset_label, experiment_name, csv_path).
    Skips any results.csv that lives inside an eval_* subfolder.
    """
    found = []
    for folder_name, dataset_label in DATASET_GROUPS.items():
        dataset_dir = results_root / folder_name
        if not dataset_dir.is_dir():
            continue
        for exp_dir in sorted(dataset_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            csv_path = exp_dir / "results.csv"
            if not csv_path.is_file():
                continue
            # Skip eval_* sub-runs (single-subset partial results)
            if exp_dir.name.startswith("eval_"):
                continue
            found.append((dataset_label, exp_dir.name, csv_path))
    return found


def read_results_csv(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Returns {row_name: {metric: value}} for every row in the CSV.
    Row names are normalised: strip dataset prefix (e.g. "mvtec_bottle" → "bottle").
    "Mean" is kept as-is.
    """
    data = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row["Row Names"].strip()
            # Normalise: remove leading "<dataset>_" prefix
            name = raw_name
            for prefix in ("mvtec_", "visa_"):
                if raw_name.lower().startswith(prefix):
                    name = raw_name[len(prefix):]
                    break
            values = {}
            for metric in METRICS:
                try:
                    values[metric] = float(row[metric])
                except (KeyError, ValueError):
                    values[metric] = float("nan")
            data[name] = values
    return data


def build_output(all_entries: List[Tuple[str, str, dict]]) -> Tuple[List[str], List[dict]]:
    """
    Builds (column_headers, rows) for the final CSV.

    Column order:
        dataset, experiment,
        instance_auroc_Mean,      instance_auroc_<subset…>,
        full_pixel_auroc_Mean,    full_pixel_auroc_<subset…>,
        anomaly_pixel_auroc_Mean, anomaly_pixel_auroc_<subset…>

    Subsets are the union of all subset names seen, sorted alphabetically,
    with Mean always first within each metric group.
    """
    # Collect all subset names (excluding "Mean") across all experiments
    all_subsets: Set[str] = set()
    for _, _, data in all_entries:
        all_subsets.update(k for k in data if k != "Mean")
    subsets_sorted = sorted(all_subsets)

    # Build column list: Mean first, then alphabetical subsets
    ordered_subsets = ["Mean"] + subsets_sorted

    headers = ["dataset", "experiment"]
    for metric in METRICS:
        for subset in ordered_subsets:
            headers.append(f"{metric}_{subset}")

    rows = []
    for dataset, experiment, data in all_entries:
        row = {"dataset": dataset, "experiment": experiment}
        for metric in METRICS:
            for subset in ordered_subsets:
                col = f"{metric}_{subset}"
                val = data.get(subset, {}).get(metric, float("nan"))
                row[col] = f"{val:.6f}" if val == val else ""  # nan → empty
        rows.append(row)

    return headers, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_root",
        default=None,
        help="Path to the results/ directory. Defaults to <repo>/results/",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Defaults to <results_root>/aggregated_results.csv",
    )
    args = parser.parse_args()

    # Resolve results root
    if args.results_root:
        results_root = Path(args.results_root)
    else:
        results_root = Path(__file__).parent.parent / "results"

    if not results_root.is_dir():
        raise SystemExit(f"Results directory not found: {results_root}")

    out_path = Path(args.out) if args.out else results_root / "aggregated_results.csv"

    # Discover all results.csv files
    entries_raw = find_results_csvs(results_root)
    if not entries_raw:
        raise SystemExit("No results.csv files found.")

    print(f"Found {len(entries_raw)} experiments:")
    for dataset, exp, path in entries_raw:
        print(f"  [{dataset}]  {exp}")

    # Parse each CSV
    all_entries: List[Tuple[str, str, dict]] = []
    for dataset, exp, csv_path in entries_raw:
        data = read_results_csv(csv_path)
        if "Mean" not in data:
            print(f"  WARNING: no Mean row in {csv_path}, skipping.")
            continue
        all_entries.append((dataset, exp, data))

    # Sort: group by dataset (MVTecAD before VisA), then by mean instance_auroc desc
    DATASET_ORDER = {"MVTecAD": 0, "VisA": 1}
    all_entries.sort(
        key=lambda e: (
            DATASET_ORDER.get(e[0], 99),
            -e[2].get("Mean", {}).get("instance_auroc", 0.0),
        )
    )

    # Build and write output
    headers, rows = build_output(all_entries)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAggregated results written to: {out_path}")
    print(f"  {len(rows)} experiments  ×  {len(headers) - 2} metric columns")


if __name__ == "__main__":
    main()
