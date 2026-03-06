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

import yaml


METRICS = ["instance_auroc", "full_pixel_auroc", "anomaly_pixel_auroc"]

DATASET_GROUPS = {
    "MVTecAD_Results": "MVTecAD",
    "VisA_Results":    "VisA",
}

# Folder-name prefixes that identify sweep output directories
SWEEP_PREFIXES = {
    "MVTecAD_Sweep_": "MVTecAD",
    "VisA_Sweep_":    "VisA",
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


def find_sweep_scores(results_root: Path) -> List[Tuple[str, str, Path]]:
    """
    Returns list of (dataset_label, experiment_name, scores_yaml_path) for every
    sweep trial that has a scores.yaml file.

    Sweep folders are identified by their prefix (e.g. VisA_Sweep_*, MVTecAD_Sweep_*).
    Experiment name is formatted as  "[StudyName] TrialName"  to distinguish
    sweep trials from regular runs in the output CSV.
    """
    found = []
    for folder in sorted(results_root.iterdir()):
        if not folder.is_dir():
            continue
        dataset_label = None
        study_name    = None
        for prefix, label in SWEEP_PREFIXES.items():
            if folder.name.startswith(prefix):
                dataset_label = label
                study_name    = folder.name[len(prefix):]
                break
        if dataset_label is None:
            continue
        for trial_dir in sorted(folder.iterdir()):
            if not trial_dir.is_dir():
                continue
            scores_path = trial_dir / "scores.yaml"
            if scores_path.is_file():
                exp_name = f"[{study_name}] {trial_dir.name}"
                found.append((dataset_label, exp_name, scores_path))
    return found


def read_scores_yaml(yaml_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Parses a sweep trial's scores.yaml into the same format as read_results_csv().

    scores.yaml structure:
        candle: 0.9512
        cashew: 0.9801
        ...
        mean_auroc: 0.9623

    Returns {subset: {metric: value}}.  Only instance_auroc is present;
    pixel metrics are left absent (will render as empty in the CSV).
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    data = {}
    for key, val in raw.items():
        subset = "Mean" if key == "mean_auroc" else key
        try:
            data[subset] = {"instance_auroc": float(val)}
        except (TypeError, ValueError):
            pass
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

    all_entries: List[Tuple[str, str, dict]] = []

    # ── Regular results (results.csv) ─────────────────────────────────────────
    entries_raw = find_results_csvs(results_root)
    if entries_raw:
        print(f"Found {len(entries_raw)} regular experiment(s):")
        for dataset, exp, path in entries_raw:
            print(f"  [{dataset}]  {exp}")
        for dataset, exp, csv_path in entries_raw:
            data = read_results_csv(csv_path)
            if "Mean" not in data:
                print(f"  WARNING: no Mean row in {csv_path}, skipping.")
                continue
            all_entries.append((dataset, exp, data))
    else:
        print("No regular results.csv files found.")

    # ── Sweep trials (scores.yaml) ────────────────────────────────────────────
    sweep_raw = find_sweep_scores(results_root)
    if sweep_raw:
        print(f"\nFound {len(sweep_raw)} sweep trial(s):")
        for dataset, exp, path in sweep_raw:
            print(f"  [{dataset}]  {exp}")
        for dataset, exp, yaml_path in sweep_raw:
            data = read_scores_yaml(yaml_path)
            if "Mean" not in data:
                print(f"  WARNING: no mean_auroc in {yaml_path}, skipping.")
                continue
            all_entries.append((dataset, exp, data))
    else:
        print("No sweep scores.yaml files found.")

    if not all_entries:
        raise SystemExit("Nothing to aggregate.")

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
