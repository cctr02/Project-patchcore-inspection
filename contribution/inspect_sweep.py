"""
inspect_sweep.py — Human-readable viewer for PatchCore sweep SQLite databases.

Usage
-----
  # Auto-locate the .db from study_name (looks in results/*_Sweep_<name>/)
  python contribution/inspect_sweep.py --study_name DINOv2B_VisA_Pilot

  # Point directly to a .db file
  python contribution/inspect_sweep.py --db results/VisA_Sweep_DINOv2B_VisA_Pilot/DINOv2B_VisA_Pilot.db

  # Show all trials (not just top-N)
  python contribution/inspect_sweep.py --study_name DINOv2B_VisA_Pilot --top 0

  # Show block-score analysis (for layer_sampling sweeps)
  python contribution/inspect_sweep.py --study_name DINOv2B_VisA_Pilot --blocks
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import List, Optional

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESULTS_ROOT = os.path.join(_REPO_ROOT, "results")

# ──────────────────────────────────────────────────────────────────────────────
# DB discovery
# ──────────────────────────────────────────────────────────────────────────────

def _find_db(study_name: str) -> str:
    """Search results/*_Sweep_<study_name>/<study_name>.db"""
    for entry in os.listdir(_RESULTS_ROOT):
        if entry.endswith(f"_Sweep_{study_name}"):
            db = os.path.join(_RESULTS_ROOT, entry, f"{study_name}.db")
            if os.path.isfile(db):
                return db
    raise FileNotFoundError(
        f"Could not find a .db for study '{study_name}' under {_RESULTS_ROOT}.\n"
        f"Pass --db <path> explicitly."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_STATE_LABEL = {
    optuna.trial.TrialState.COMPLETE: "COMPLETE",
    optuna.trial.TrialState.PRUNED:   "PRUNED  ",
    optuna.trial.TrialState.FAIL:     "FAIL    ",
    optuna.trial.TrialState.RUNNING:  "RUNNING ",
    optuna.trial.TrialState.WAITING:  "WAITING ",
}


def _trial_auroc(trial: optuna.trial.FrozenTrial) -> float:
    """Return the best reported value (intermediate or final) for a trial."""
    if trial.value is not None:
        return trial.value
    if trial.intermediate_values:
        return max(trial.intermediate_values.values())
    return float("nan")


def _layer_key(params: dict) -> str:
    """Reconstruct the layer key from block_score_* params (layer_sampling mode)."""
    scores = {k: v for k, v in params.items() if k.startswith("block_score_")}
    if not scores:
        return params.get("layer_key", "?")
    n = params.get("n_layers", 2)
    selected = sorted(scores, key=lambda k: -scores[k])[:n]
    indices = sorted(int(k.split("_")[-1]) for k in selected)
    return "L" + "-".join(str(i) for i in indices)


def _short_params(params: dict) -> str:
    """One-line summary of the non-block-score params."""
    parts = []
    lk = _layer_key(params)
    parts.append(lk)
    cf = params.get("coreset_fraction")
    if cf is not None:
        parts.append(f"P{int(round(cf * 100)):03d}")
    ps = params.get("patchsize")
    if ps is not None:
        parts.append(f"ps={ps}")
    nn = params.get("anomaly_scorer_num_nn")
    if nn is not None:
        parts.append(f"nn={nn}")
    pre = params.get("pretrain_embed_dimension")
    tgt = params.get("target_embed_dimension")
    if pre and tgt:
        parts.append(f"D{pre}-{tgt}")
    ri = params.get("resize_imagesize")
    if ri:
        parts.append(f"sz={ri}")
    return "  ".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# Display sections
# ──────────────────────────────────────────────────────────────────────────────

def _sep(char="─", width=72):
    print(char * width)


def print_summary(study: optuna.Study):
    trials = study.trials
    by_state = defaultdict(list)
    for t in trials:
        by_state[t.state].append(t)

    n_complete = len(by_state[optuna.trial.TrialState.COMPLETE])
    n_pruned   = len(by_state[optuna.trial.TrialState.PRUNED])
    n_fail     = len(by_state[optuna.trial.TrialState.FAIL])
    n_running  = len(by_state[optuna.trial.TrialState.RUNNING])

    _sep("═")
    print(f"  STUDY : {study.study_name}")
    _sep("═")
    print(f"  Total trials   : {len(trials)}")
    print(f"  Complete       : {n_complete}")
    print(f"  Pruned         : {n_pruned}")
    print(f"  Failed         : {n_fail}")
    if n_running:
        print(f"  Running        : {n_running}  (interrupted mid-trial?)")

    if n_complete:
        aurocs = [t.value for t in by_state[optuna.trial.TrialState.COMPLETE]]
        print(f"\n  Mean AUROC — best : {max(aurocs):.4f}")
        print(f"             median : {float(np.median(aurocs)):.4f}")
        print(f"             worst  : {min(aurocs):.4f}")
    print()


def print_best(study: optuna.Study):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("  No completed trials yet.\n")
        return

    best = study.best_trial
    _sep()
    print(f"  BEST TRIAL  #{best.number}  —  Mean AUROC {best.value:.4f}")
    _sep()

    # Layers
    print(f"  Layers      : {_layer_key(best.params)}")

    # Key params (non-block-score)
    skip = {"resize_imagesize", "n_layers"}
    for k, v in best.params.items():
        if k.startswith("block_score_") or k in skip:
            continue
        print(f"  {k:<30}: {v}")

    # resize/imagesize
    ri = best.params.get("resize_imagesize", "")
    if ri:
        print(f"  {'resize_imagesize':<30}: {ri}")

    # Block scores (if layer_sampling)
    scores = {k: v for k, v in best.params.items() if k.startswith("block_score_")}
    if scores:
        print()
        print("  Block scores (higher = more selected by TPE):")
        for k in sorted(scores, key=lambda k: -scores[k]):
            idx = k.split("_")[-1]
            bar = "█" * int(scores[k] * 20)
            print(f"    block {idx:>2}  {scores[k]:.4f}  {bar}")

    # trial_name user attr
    tn = best.user_attrs.get("trial_name", "")
    if tn:
        print(f"\n  Trial name  : {tn}")
    print()


def print_trials_table(study: optuna.Study, top_n: int = 20):
    completed = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: -t.value,
    )
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    _sep()
    label = f"ALL {len(completed)}" if top_n == 0 else f"TOP {min(top_n, len(completed))}"
    print(f"  COMPLETED TRIALS ({label})")
    _sep()

    show = completed if top_n == 0 else completed[:top_n]
    if not show:
        print("  (none)\n")
    else:
        rank_width = len(str(len(show)))
        for rank, t in enumerate(show, 1):
            params_str = _short_params(t.params)
            print(f"  #{rank:<{rank_width}}  [{t.number:>3}]  {t.value:.4f}  {params_str}")
        print()

    if pruned:
        _sep()
        print(f"  PRUNED TRIALS ({len(pruned)})")
        _sep()
        for t in sorted(pruned, key=lambda t: t.number):
            best_iv = max(t.intermediate_values.values()) if t.intermediate_values else float("nan")
            n_steps = max(t.intermediate_values.keys()) if t.intermediate_values else 0
            params_str = _short_params(t.params) if t.params else "(no params)"
            print(f"  [{t.number:>3}]  best={best_iv:.4f}  after {n_steps} class(es)  {params_str}")
        print()

    if failed:
        _sep()
        print(f"  FAILED TRIALS ({len(failed)})")
        _sep()
        for t in sorted(failed, key=lambda t: t.number):
            print(f"  [{t.number:>3}]  {t.params}")
        print()


def print_block_analysis(study: optuna.Study, top_n: int = 10):
    """
    For layer_sampling sweeps: show which blocks appear most often in
    the top-N trials and how their presence correlates with AUROC.
    """
    completed = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: -t.value,
    )
    if not completed:
        print("  No completed trials for block analysis.\n")
        return

    # Check this is a layer_sampling sweep
    has_blocks = any(
        k.startswith("block_score_") for k in completed[0].params
    )
    if not has_blocks:
        print("  Not a layer_sampling sweep — no block analysis available.\n")
        return

    # Collect which blocks are selected in each trial
    block_stats = defaultdict(lambda: {"selected": [], "auroc_when_selected": [],
                                        "auroc_when_not": []})
    all_blocks = set()

    for t in completed:
        scores = {k: v for k, v in t.params.items() if k.startswith("block_score_")}
        n = t.params.get("n_layers", 2)
        selected = set(
            int(k.split("_")[-1])
            for k in sorted(scores, key=lambda k: -scores[k])[:n]
        )
        all_blocks.update(int(k.split("_")[-1]) for k in scores)
        for b in all_blocks:
            if b in selected:
                block_stats[b]["selected"].append(t.number)
                block_stats[b]["auroc_when_selected"].append(t.value)
            else:
                block_stats[b]["auroc_when_not"].append(t.value)

    _sep()
    print(f"  BLOCK ANALYSIS  ({len(completed)} completed trials)")
    _sep()
    print(f"  {'Block':<8}  {'Freq':>5}  {'Avg AUROC if selected':>22}  "
          f"{'Avg AUROC if absent':>20}  {'Δ':>8}")
    _sep("-")

    rows = []
    for b in sorted(all_blocks):
        s = block_stats[b]
        freq = len(s["auroc_when_selected"]) / len(completed)
        avg_sel = float(np.mean(s["auroc_when_selected"])) if s["auroc_when_selected"] else float("nan")
        avg_not = float(np.mean(s["auroc_when_not"])) if s["auroc_when_not"] else float("nan")
        delta   = avg_sel - avg_not if (avg_sel == avg_sel and avg_not == avg_not) else float("nan")
        rows.append((b, freq, avg_sel, avg_not, delta))

    # Sort by delta desc (most beneficial block first)
    rows.sort(key=lambda r: -(r[4] if r[4] == r[4] else -999))

    for b, freq, avg_sel, avg_not, delta in rows:
        delta_str = f"{delta:+.4f}" if delta == delta else "   N/A"
        print(f"  block {b:<4}  {freq:>5.0%}  {avg_sel:>22.4f}  {avg_not:>20.4f}  {delta_str:>8}")

    print()

    # Top-N trials: which blocks are most common?
    top = completed[:top_n]
    if len(top) > 1:
        _sep("-")
        print(f"  Block frequency in TOP {len(top)} trials vs ALL trials:")
        _sep("-")
        for b in sorted(all_blocks):
            top_freq = sum(
                1 for t in top
                if b in set(
                    int(k.split("_")[-1])
                    for k in sorted(
                        {k: v for k, v in t.params.items() if k.startswith("block_score_")},
                        key=lambda k: -t.params[k]
                    )[:t.params.get("n_layers", 2)]
                )
            ) / len(top)
            all_freq = block_stats[b]["selected"].__len__() / len(completed)
            bar_top = "█" * int(top_freq * 20)
            print(f"  block {b:<4}  top={top_freq:>5.0%}  all={all_freq:>5.0%}  {bar_top}")
        print()


def print_param_distributions(study: optuna.Study):
    """Show value distribution for non-block-score categorical/float params."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return

    _sep()
    print("  PARAMETER DISTRIBUTIONS (completed trials)")
    _sep()

    # Gather all non-block-score params
    param_keys = sorted(set(
        k for t in completed for k in t.params
        if not k.startswith("block_score_") and k != "n_layers"
    ))

    for key in param_keys:
        vals = [t.params[key] for t in completed if key in t.params]
        auroc_by_val = defaultdict(list)
        for t in completed:
            if key in t.params:
                auroc_by_val[t.params[key]].append(t.value)

        if isinstance(vals[0], (int, float)) and len(set(vals)) > 5:
            # Continuous — show min/mean/max
            arr = np.array(vals, dtype=float)
            print(f"  {key:<35}  min={arr.min():.4f}  mean={arr.mean():.4f}  max={arr.max():.4f}")
        else:
            # Categorical — show each value with avg AUROC
            print(f"  {key}:")
            for val in sorted(auroc_by_val, key=lambda v: -np.mean(auroc_by_val[v])):
                avg = float(np.mean(auroc_by_val[val]))
                cnt = len(auroc_by_val[val])
                print(f"    {str(val):<20}  n={cnt:>3}  avg_auroc={avg:.4f}")
    print()

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--study_name", type=str,
                       help="Study name (auto-locates the .db in results/).")
    group.add_argument("--db", type=str,
                       help="Direct path to the Optuna SQLite .db file.")

    parser.add_argument("--top", type=int, default=20,
                        help="Show top-N completed trials. 0 = show all. (default: 20)")
    parser.add_argument("--blocks", action="store_true",
                        help="Show per-block analysis (layer_sampling sweeps only).")
    parser.add_argument("--params", action="store_true",
                        help="Show parameter distribution analysis.")
    parser.add_argument("--all", dest="show_all", action="store_true",
                        help="Equivalent to --blocks --params --top 0.")
    args = parser.parse_args()

    if args.show_all:
        args.blocks = True
        args.params = True
        args.top    = 0

    # ── Load study ────────────────────────────────────────────────────────────
    if args.db:
        db_path    = os.path.abspath(args.db)
        study_name = os.path.splitext(os.path.basename(db_path))[0]
    else:
        db_path    = _find_db(args.study_name)
        study_name = args.study_name

    storage_url = f"sqlite:///{db_path}"
    print(f"\nLoading: {db_path}")

    study = optuna.load_study(study_name=study_name, storage=storage_url)

    # ── Sections ──────────────────────────────────────────────────────────────
    print_summary(study)
    print_best(study)
    print_trials_table(study, top_n=args.top)

    if args.blocks:
        print_block_analysis(study, top_n=max(args.top, 5) if args.top else 10)

    if args.params:
        print_param_distributions(study)


if __name__ == "__main__":
    main()
