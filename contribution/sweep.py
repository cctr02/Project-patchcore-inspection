"""
Bayesian hyperparameter sweep for PatchCore (config-driven).

Everything is specified in a YAML config file; the only required CLI argument
is --study_name (or --config to point to a custom file).

Default config lookup:
    contribution/sweep_configs/{study_name}.yaml

Usage
-----
  # New sweep (reads contribution/sweep_configs/ConvNeXtV2B_FCMAE_Pilot.yaml)
  python contribution/sweep.py --study_name ConvNeXtV2B_FCMAE_Pilot

  # Resume an interrupted sweep (SQLite keeps completed trials)
  python contribution/sweep.py --study_name ConvNeXtV2B_FCMAE_Pilot

  # Custom config path
  python contribution/sweep.py --study_name MySweep --config path/to/config.yaml

Folder layout (from results_root in config)
-------------------------------------------
  results/{sweep_folder_prefix}_Sweep_{study_name}/
    {study_name}.db          <- Optuna SQLite (resumable)
    best_trial.yaml          <- written after sweep completes
    IM224_ConvNeXtV2B_FCMAE_L1-2_P01_D1024-1024_PS-3_AN-1_S0/
      config.yaml            <- trial hyperparameters
      scores.yaml            <- per-class AUROC + mean
"""

import gc
import logging
import os
import sys
import types
from typing import Optional

import click
import numpy as np
import optuna
import torch
import yaml

# Make project root importable from anywhere
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from contribution.backbones_extension import _CONVNEXTV2_TIMM_NAMES, _DINOV2_TIMM_NAMES

LOGGER = logging.getLogger(__name__)

# Default location for sweep config files
_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "sweep_configs")

# Maps dataset key → folder prefix used in results path
_DATASET_PREFIX = {
    "mvtec": "MVTecAD",
    "visa":  "VisA",
}

# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(study_name: str, config_path: Optional[str]) -> dict:
    if config_path is None:
        config_path = os.path.join(_CONFIGS_DIR, f"{study_name}.yaml")
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Sweep config not found: {config_path}\n"
            f"Create it or pass --config <path>."
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Inject study_name so the rest of the code always has it
    cfg.setdefault("study_name", study_name)
    return cfg

# ──────────────────────────────────────────────────────────────────────────────
# Trial naming
# ──────────────────────────────────────────────────────────────────────────────

def _trial_name(params: dict, cfg: dict) -> str:
    """
    IM{imagesize}_{backbone_short}_{layer_key}_P{pct:03d}_D{pre}-{tgt}_PS-{ps}_AN-{nn}_S{seed}
    Example: IM224_ConvNeXtV2B_FCMAE_L1-2_P001_D1024-1024_PS-3_AN-1_S0
    """
    pct_int = max(1, int(round(params["coreset_fraction"] * 100)))
    return (
        f"IM{params['imagesize']}"
        f"_{cfg['backbone_short']}"
        f"_{params['layer_key']}"
        f"_P{pct_int:03d}"
        f"_D{params['pretrain_embed_dimension']}-{params['target_embed_dimension']}"
        f"_PS-{params['patchsize']}"
        f"_AN-{params['anomaly_scorer_num_nn']}"
        f"_S{cfg['seed']}"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Dynamic search-space sampling
# ──────────────────────────────────────────────────────────────────────────────

def _sample_params(trial: optuna.Trial, cfg: dict) -> dict:
    """
    Reads cfg["search_space"] and calls the appropriate trial.suggest_* methods.

    Special keys handled separately (not dispatched generically):

      resize_imagesize_pairs : list of [resize, imagesize] pairs; suggested as a
                               categorical.  Use for DINOv2 (imagesize must be a
                               multiple of 14).  Takes priority over resize/offset.
      resize                 : categorical choices; imagesize = resize - imagesize_offset.
      imagesize_offset       : fixed int (default 32); NOT suggested by Optuna.

      layer_combos           : dict {key: [layers]}; key suggested as categorical.
                               TPE treats the combo as a single opaque token.

      layer_sampling         : dynamic mode — each candidate block gets its own
                               float score; TPE can learn "block N is useful"
                               independently and bias future combos accordingly.
                               Fields:
                                 candidates   : list of block indices, e.g. [3..11]
                                 n_layers     : list of allowed combo sizes, e.g. [2, 3]
                                 layer_prefix : str, e.g. "blocks" or "stages"
    """
    ss = cfg["search_space"]
    params: dict = {}

    # ── resize + imagesize ─────────────────────────────────────────────────────
    if "resize_imagesize_pairs" in ss:
        pairs = ss["resize_imagesize_pairs"]
        pair_keys = ["{}x{}".format(r, i) for r, i in pairs]
        chosen = trial.suggest_categorical("resize_imagesize", pair_keys)
        idx = pair_keys.index(chosen)
        params["resize"]    = pairs[idx][0]
        params["imagesize"] = pairs[idx][1]
    else:
        resize = trial.suggest_categorical("resize", ss["resize"]["choices"])
        offset = ss.get("imagesize_offset", 32)
        params["resize"]    = resize
        params["imagesize"] = resize - offset

    # ── layer selection ────────────────────────────────────────────────────────
    if "layer_sampling" in ss:
        # Dynamic mode: TPE assigns a score to each candidate block independently.
        # High-scoring blocks are selected more often → TPE learns "block N matters".
        ls         = ss["layer_sampling"]
        candidates = ls["candidates"]           # e.g. [3, 4, 5, 6, 7, 8, 9, 10, 11]
        n_choices  = ls["n_layers"]             # e.g. [2, 3]
        prefix     = ls.get("layer_prefix", "blocks")

        n_layers = trial.suggest_categorical("n_layers", n_choices)

        # Each candidate block gets its own continuous score in [0, 1].
        # TPE will learn to push scores high for blocks that improve AUROC.
        scores = {
            b: trial.suggest_float("block_score_{}".format(b), 0.0, 1.0)
            for b in candidates
        }
        # Select the n_layers blocks with the highest scores, sort ascending.
        selected = sorted(candidates, key=lambda b: -scores[b])[:n_layers]
        selected.sort()

        params["layer_key"]              = "L" + "-".join(str(b) for b in selected)
        params["layers_to_extract_from"] = ["{}.{}".format(prefix, b) for b in selected]

    elif "layer_combos" in ss:
        # Fixed mode: combo is a single opaque categorical.
        # Faster convergence when the good combos are known in advance.
        layer_combos = ss["layer_combos"]
        layer_key    = trial.suggest_categorical("layer_key", list(layer_combos.keys()))
        params["layer_key"]              = layer_key
        params["layers_to_extract_from"] = layer_combos[layer_key]

    else:
        raise ValueError(
            "search_space must contain either 'layer_sampling' or 'layer_combos'."
        )

    # ── all other search-space params (generic dispatch) ───────────────────────
    _skip = {
        "resize", "imagesize_offset", "resize_imagesize_pairs",
        "layer_combos", "layer_sampling",
    }
    for name, spec in ss.items():
        if name in _skip:
            continue
        t = spec["type"]
        if t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif t == "float_log":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif t == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            raise ValueError(
                "Unknown search_space type '{}' for param '{}'".format(t, name)
            )

    return params

# ──────────────────────────────────────────────────────────────────────────────
# Dataset factory (mvtec / visa)
# ──────────────────────────────────────────────────────────────────────────────

def _make_datasets(cfg: dict, classname: str, resize: int, imagesize: int, seed: int):
    dataset_key     = cfg["dataset"]
    data_path       = cfg["data_path"]
    train_val_split = cfg.get("train_val_split", 1.0)

    if dataset_key == "mvtec":
        import patchcore.datasets.mvtec as ds_lib
        train_ds = ds_lib.MVTecDataset(
            data_path, classname=classname,
            resize=resize, imagesize=imagesize,
            split=ds_lib.DatasetSplit.TRAIN,
            seed=seed, train_val_split=train_val_split,
        )
        test_ds = ds_lib.MVTecDataset(
            data_path, classname=classname,
            resize=resize, imagesize=imagesize,
            split=ds_lib.DatasetSplit.TEST,
            seed=seed,
        )

    elif dataset_key == "visa":
        import contribution.visa as ds_lib
        train_ds = ds_lib.VisADataset(
            data_path, classname=classname,
            resize=resize, imagesize=imagesize,
            split=ds_lib.DatasetSplit.TRAIN,
            train_val_split=train_val_split,
        )
        test_ds = ds_lib.VisADataset(
            data_path, classname=classname,
            resize=resize, imagesize=imagesize,
            split=ds_lib.DatasetSplit.TEST,
            train_val_split=train_val_split,
        )

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. Supported: mvtec, visa."
        )

    return train_ds, test_ds

# ──────────────────────────────────────────────────────────────────────────────
# Backbone + NN-method factory (ConvNeXt V2 / standard)
# ──────────────────────────────────────────────────────────────────────────────

def _make_backbone_and_nn(cfg: dict):
    backbone_name     = cfg["backbone"]
    faiss_on_gpu      = cfg.get("faiss_on_gpu", False)
    faiss_num_workers = cfg.get("faiss_num_workers", 4)

    dino_timm_name = _DINOV2_TIMM_NAMES.get(backbone_name)
    timm_name      = _CONVNEXTV2_TIMM_NAMES.get(backbone_name)

    if dino_timm_name is not None:
        # DINOv2: DINOv2Aggregator (patchcore.py detects backbone.dino_timm_name) + CosineNN.
        backbone  = types.SimpleNamespace(
            name=backbone_name, seed=None, dino_timm_name=dino_timm_name
        )
        nn_method = patchcore.common.CosineNN(
            on_gpu=faiss_on_gpu, num_workers=faiss_num_workers
        )
    elif timm_name is not None:
        # ConvNeXt V2: FeaturesOnlyAggregator + CosineNN.
        backbone  = types.SimpleNamespace(
            name=backbone_name, seed=None, timm_name=timm_name
        )
        nn_method = patchcore.common.CosineNN(
            on_gpu=faiss_on_gpu, num_workers=faiss_num_workers
        )
    else:
        # Standard backbone (WideResNet, etc.): hook-based aggregator + L2 FAISS.
        backbone = patchcore.backbones.load(backbone_name)
        backbone.name = backbone_name
        backbone.seed = None
        nn_method = patchcore.common.FaissNN(
            on_gpu=faiss_on_gpu, num_workers=faiss_num_workers
        )

    return backbone, nn_method

# ──────────────────────────────────────────────────────────────────────────────
# Single-class training + evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _run_one_class(
    classname: str,
    params: dict,
    cfg: dict,
    device: torch.device,
) -> float:
    """Train PatchCore on one class and return image-level AUROC."""

    seed        = cfg["seed"]
    batch_size  = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    patchcore.utils.fix_seeds(seed, device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds, test_ds = _make_datasets(
        cfg, classname, params["resize"], params["imagesize"], seed
    )
    pin = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    # ── Backbone / NN method ──────────────────────────────────────────────────
    backbone, nn_method = _make_backbone_and_nn(cfg)

    # ── Coreset sampler ───────────────────────────────────────────────────────
    coreset_fraction = params["coreset_fraction"]
    if coreset_fraction >= 1.0:
        sampler = patchcore.sampler.IdentitySampler()
    else:
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
            percentage=coreset_fraction, device=device
        )

    # ── Build + train PatchCore ───────────────────────────────────────────────
    input_shape = train_ds.imagesize   # (3, H, W) tuple
    pc = patchcore.patchcore.PatchCore(device)
    pc.load(
        backbone=backbone,
        layers_to_extract_from=params["layers_to_extract_from"],
        device=device,
        input_shape=input_shape,
        pretrain_embed_dimension=params["pretrain_embed_dimension"],
        target_embed_dimension=params["target_embed_dimension"],
        patchsize=params["patchsize"],
        featuresampler=sampler,
        anomaly_scorer_num_nn=params["anomaly_scorer_num_nn"],
        nn_method=nn_method,
    )
    pc.fit(train_loader)

    # ── Evaluate (image-level AUROC only) ─────────────────────────────────────
    scores, _masks, labels_gt, _masks_gt = pc.predict(test_loader)
    scores = np.array(scores)
    rng = scores.max() - scores.min()
    scores_norm = (scores - scores.min()) / (rng if rng > 0 else 1.0)
    auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
        scores_norm, labels_gt
    )["auroc"]

    del pc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return float(auroc)

# ──────────────────────────────────────────────────────────────────────────────
# Optuna objective factory
# ──────────────────────────────────────────────────────────────────────────────

def _pruning_stats(study: optuna.Study, step: int) -> dict:
    """
    Compute comparison stats between the current trial's intermediate mean and
    the distribution of completed trials at the same step.

    Returns a dict with keys: median, mean, min, max, n_refs, and their values.
    """
    refs = [
        t.intermediate_values[step]
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and step in t.intermediate_values
    ]
    if not refs:
        return {"n_refs": 0, "median": float("nan"), "mean": float("nan"),
                "min": float("nan"), "max": float("nan")}
    arr = np.array(refs)
    return {
        "n_refs": len(refs),
        "median": float(np.median(arr)),
        "mean":   float(np.mean(arr)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
    }


def _append_trial_report(
    report_path: str,
    study_name: str,
    trial_number: int,
    trial_name: str,
    status: str,                   # "COMPLETE" | "PRUNED" | "FAILED"
    class_aurocs: dict,
    pruned_at_cls: str = None,     # last class when pruned
    prune_stats: dict = None,      # output of _pruning_stats
    current_mean: float = None,    # running mean when pruned
    failed_cls: str = None,
    error: str = None,
):
    """Append a trial section to the human-readable Markdown report."""
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []

    # ── Header (only on first write) ─────────────────────────────────────────
    if not os.path.isfile(report_path):
        lines += [
            f"# Sweep Report — {study_name}",
            "",
            "Each trial is appended as it finishes (complete, pruned, or failed).",
            "",
        ]

    # ── Trial section ─────────────────────────────────────────────────────────
    icon = {"COMPLETE": "✅", "PRUNED": "✂️ ", "FAILED": "❌"}.get(status, "?")
    lines += [
        f"---",
        f"## Trial {trial_number} — {status} {icon}  `{now}`",
        f"**Name**: `{trial_name}`",
        "",
    ]

    if status == "COMPLETE":
        mean = float(np.mean(list(class_aurocs.values()))) if class_aurocs else float("nan")
        lines += [
            f"**Mean AUROC**: **{mean:.4f}** ({len(class_aurocs)} classes)",
            "",
            "| Class | AUROC |",
            "|-------|------:|",
        ]
        for cls, auc in sorted(class_aurocs.items()):
            lines.append(f"| {cls} | {auc:.4f} |")

    elif status == "PRUNED" and pruned_at_cls is not None:
        n_done = len(class_aurocs)
        lines += [
            f"**Stopped after**: {n_done} class(es) (pruned at `{pruned_at_cls}`)",
            f"**Running mean**: {current_mean:.4f}",
            "",
        ]
        if prune_stats and prune_stats["n_refs"] > 0:
            ps = prune_stats
            delta = current_mean - ps["median"]
            pct = (delta / ps["median"] * 100) if ps["median"] else 0.0
            lines += [
                "**Pruning criterion** (MedianPruner — running mean vs completed trials at same step):",
                "",
                f"| Metric | Value |",
                f"|--------|------:|",
                f"| Running mean (this trial) | {current_mean:.4f} |",
                f"| Median @ step {n_done}     | {ps['median']:.4f} |",
                f"| Mean @ step {n_done}       | {ps['mean']:.4f} |",
                f"| Best @ step {n_done}       | {ps['max']:.4f} |",
                f"| Worst @ step {n_done}      | {ps['min']:.4f} |",
                f"| Δ vs median                | {delta:+.4f} ({pct:+.1f}%) |",
                f"| Reference trials           | {ps['n_refs']} |",
                "",
                f"> Running mean **{current_mean:.4f}** is "
                f"{'below' if delta < 0 else 'above'} the median "
                f"**{ps['median']:.4f}** by {abs(delta):.4f} "
                f"({abs(pct):.1f}%) → pruned.",
            ]
        else:
            lines.append("> Pruned (no reference trials available at this step).")

        if class_aurocs:
            lines += [
                "",
                "**Partial results:**",
                "",
                "| Class | AUROC |",
                "|-------|------:|",
            ]
            for cls, auc in sorted(class_aurocs.items()):
                lines.append(f"| {cls} | {auc:.4f} |")

    elif status == "FAILED":
        lines += [
            f"**Failed at class**: `{failed_cls}`",
            f"**Error**: `{error}`",
        ]
        if class_aurocs:
            lines += [
                "",
                "**Partial results before failure:**",
                "",
                "| Class | AUROC |",
                "|-------|------:|",
            ]
            for cls, auc in sorted(class_aurocs.items()):
                lines.append(f"| {cls} | {auc:.4f} |")

    lines.append("")

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _make_objective(cfg: dict, sweep_dir: str, device: torch.device,
                    study: optuna.Study):
    classes     = cfg["classes"]
    report_path = os.path.join(sweep_dir, "report.md")

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, cfg)

        # Add fixed metadata needed by trial_name / YAML
        params["backbone"] = cfg["backbone"]
        params["classes"]  = classes
        params["seed"]     = cfg["seed"]

        trial_name = _trial_name(params, cfg)
        trial_dir  = os.path.join(sweep_dir, trial_name)
        os.makedirs(trial_dir, exist_ok=True)

        # Persist trial_name so best_trial.yaml can retrieve it without reconstruction.
        trial.set_user_attr("trial_name", trial_name)

        # Save config immediately (survives crashes)
        with open(os.path.join(trial_dir, "config.yaml"), "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)

        LOGGER.info(
            "[Trial %d] %s  coreset=%.4f",
            trial.number, trial_name, params["coreset_fraction"],
        )

        # ── Run on each class ────────────────────────────────────────────────
        class_aurocs = {}
        for cls in classes:
            LOGGER.info("  -> %s", cls)
            try:
                auroc = _run_one_class(cls, params, cfg, device)
            except Exception as exc:
                import traceback
                err_msg = traceback.format_exc()
                LOGGER.error("  [FAILED] %s: %s", cls, exc, exc_info=True)
                with open(os.path.join(trial_dir, "scores.yaml"), "w") as f:
                    yaml.dump(
                        {**class_aurocs,
                         "failed_class": cls,
                         "error": str(exc),
                         "traceback": err_msg},
                        f, default_flow_style=False, sort_keys=False,
                    )
                _append_trial_report(
                    report_path, cfg["study_name"], trial.number, trial_name,
                    "FAILED", class_aurocs, failed_cls=cls, error=str(exc),
                )
                raise optuna.TrialPruned(f"Class {cls} failed: {exc}")

            class_aurocs[cls] = auroc
            LOGGER.info("     AUROC = %.4f", auroc)

            # Intermediate reporting lets MedianPruner cut bad trials early
            # (pruning only fires after n_warmup_steps=3 classes)
            step         = len(class_aurocs)
            running_mean = float(np.mean(list(class_aurocs.values())))
            trial.report(running_mean, step=step)

            if trial.should_prune():
                stats = _pruning_stats(study, step)
                delta = running_mean - stats["median"] if stats["n_refs"] else float("nan")
                LOGGER.warning(
                    "  [PRUNED] after %d class(es)  mean=%.4f  "
                    "median@step%d=%.4f  Δ=%+.4f  (refs=%d)",
                    step, running_mean, step,
                    stats["median"], delta, stats["n_refs"],
                )
                with open(os.path.join(trial_dir, "scores.yaml"), "w") as f:
                    yaml.dump(
                        {**class_aurocs,
                         "pruned_after_n_classes": step,
                         "pruned_at_class": cls,
                         "running_mean": running_mean,
                         "median_at_step": stats["median"],
                         "delta_vs_median": delta},
                        f, default_flow_style=False, sort_keys=False,
                    )
                _append_trial_report(
                    report_path, cfg["study_name"], trial.number, trial_name,
                    "PRUNED", class_aurocs,
                    pruned_at_cls=cls,
                    prune_stats=stats,
                    current_mean=running_mean,
                )
                raise optuna.TrialPruned("Pruned by MedianPruner.")

        mean_auroc = float(np.mean(list(class_aurocs.values())))

        with open(os.path.join(trial_dir, "scores.yaml"), "w") as f:
            yaml.dump(
                {**class_aurocs, "mean_auroc": mean_auroc},
                f, default_flow_style=False, sort_keys=False,
            )

        _append_trial_report(
            report_path, cfg["study_name"], trial.number, trial_name,
            "COMPLETE", class_aurocs,
        )
        LOGGER.info("[Trial %d] Mean AUROC = %.4f", trial.number, mean_auroc)
        return mean_auroc

    return objective

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _discover_classes(data_path: str, dataset: str) -> list:
    """
    Return sorted list of class names from data_path, filtered by dataset structure.

    VisA  : keep dirs that contain image_anno.csv  (excludes split_csv/ etc.)
    MVTec : keep dirs that contain a train/ subdir
    """
    def _is_class_dir(name: str) -> bool:
        d = os.path.join(data_path, name)
        if not os.path.isdir(d):
            return False
        if dataset == "visa":
            return os.path.isfile(os.path.join(d, "image_anno.csv"))
        else:  # mvtec and others
            return os.path.isdir(os.path.join(d, "train"))

    entries = sorted(e for e in os.listdir(data_path) if _is_class_dir(e))
    if not entries:
        raise RuntimeError(
            f"No class directories found in {data_path} for dataset='{dataset}'."
        )
    return entries


@click.command()
@click.option(
    "--study_name", required=True, type=str,
    help="Name of the Optuna study. Used to locate sweep_configs/{study_name}.yaml.",
)
@click.option(
    "--config", default=None, type=click.Path(exists=True),
    help="Override config file path (default: contribution/sweep_configs/{study_name}.yaml).",
)
def main(study_name: str, config: Optional[str]):
    """Bayesian hyperparameter sweep for PatchCore (reads all settings from YAML)."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = _load_config(study_name, config)

    # Fields that must be present
    required = ["data_path", "dataset", "backbone", "backbone_short",
                "search_space", "seed", "n_trials", "batch_size", "num_workers"]
    for key in required:
        if key not in cfg:
            raise KeyError(f"Missing required key in sweep config: '{key}'")

    # ── Resolve class list (always all classes) ───────────────────────────────
    cfg["classes"] = _discover_classes(cfg["data_path"], cfg["dataset"])

    seed        = cfg["seed"]
    n_trials    = cfg["n_trials"]
    gpu_id      = cfg.get("gpu", 0)
    results_root = cfg.get("results_root", "results")

    dataset_prefix = _DATASET_PREFIX.get(cfg["dataset"], cfg["dataset"].upper())
    sweep_dir = os.path.join(results_root, f"{dataset_prefix}_Sweep_{study_name}")
    os.makedirs(sweep_dir, exist_ok=True)

    # ── File logging (sweep.log in sweep dir, append mode for resumed sweeps) ──
    log_path = os.path.join(sweep_dir, "sweep.log")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)

    storage_url = f"sqlite:///{os.path.abspath(os.path.join(sweep_dir, study_name + '.db'))}"

    LOGGER.info("Study       : %s", study_name)
    LOGGER.info("Config      : %s", config or os.path.join(_CONFIGS_DIR, f"{study_name}.yaml"))
    LOGGER.info("Backbone    : %s  (%s)", cfg["backbone"], cfg["backbone_short"])
    LOGGER.info("Dataset     : %s  (%s)", cfg["dataset"], cfg["data_path"])
    LOGGER.info("Classes (%d): %s",
                len(cfg["classes"]),
                cfg["classes"] if len(cfg["classes"]) <= 6 else cfg["classes"][:6] + ["..."])
    ss = cfg["search_space"]
    if "layer_sampling" in ss:
        ls = ss["layer_sampling"]
        LOGGER.info(
            "Layer mode  : dynamic  candidates=%s  n_layers=%s  prefix=%s",
            ls["candidates"], ls["n_layers"], ls.get("layer_prefix", "blocks"),
        )
    else:
        LOGGER.info("Layer combos: %s", list(ss["layer_combos"].keys()))
    LOGGER.info("n_trials    : %d", n_trials)
    LOGGER.info("Sweep dir   : %s", os.path.abspath(sweep_dir))

    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu"
    )
    LOGGER.info("Device      : %s", device)

    # ── Optuna study ──────────────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    n_done = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    LOGGER.info("Already completed: %d trial(s). Running %d more.", n_done, n_trials)

    objective = _make_objective(cfg, sweep_dir, device, study)

    interrupted = False
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            catch=(Exception,),
        )
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.warning("=" * 60)
        LOGGER.warning("Sweep interrupted by user (Ctrl+C).")
        LOGGER.warning("Completed trials are saved and resumable.")

    # ── Final summary (written on normal completion AND on interrupt) ──────────
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        LOGGER.warning("No trials completed successfully — no best_trial.yaml written.")
        return

    best = study.best_trial
    LOGGER.info("=" * 60)
    if interrupted:
        LOGGER.info("Partial summary (sweep interrupted after %d completed trial(s)):",
                    len(completed))
    LOGGER.info("Best trial  #%d", best.number)
    LOGGER.info("Mean AUROC  : %.4f", best.value)
    LOGGER.info("Params      : %s", best.params)

    best_trial_name = best.user_attrs.get("trial_name", "(see best_params)")

    summary = {
        "study_name":          study_name,
        "interrupted":         interrupted,
        "n_completed_trials":  len(completed),
        "best_trial_number":   best.number,
        "best_mean_auroc":     best.value,
        "best_trial_name":     best_trial_name,
        "best_params":         dict(best.params),
    }
    summary_path = os.path.join(sweep_dir, "best_trial.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    LOGGER.info("Summary written to %s", summary_path)
    if interrupted:
        LOGGER.info("Resume with: python contribution/sweep.py --study_name %s",
                    study_name)


if __name__ == "__main__":
    main()
