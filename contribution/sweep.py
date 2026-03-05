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
    IM{imagesize}_{backbone_short}_{layer_key}_P{pct:02d}_D{pre}-{tgt}_PS-{ps}_AN-{nn}_S{seed}
    Example: IM224_ConvNeXtV2B_FCMAE_L1-2_P01_D1024-1024_PS-3_AN-1_S0
    """
    pct_int = max(1, int(round(params["coreset_pct"] * 100)))
    return (
        f"IM{params['imagesize']}"
        f"_{cfg['backbone_short']}"
        f"_{params['layer_key']}"
        f"_P{pct_int:02d}"
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

    Special keys in search_space:
      resize_imagesize_pairs : list of [resize, imagesize] pairs (for DINOv2,
                               where imagesize must be a multiple of 14). Suggested
                               as a categorical; takes priority over resize/imagesize_offset.
      resize          : categorical choices; imagesize derived as resize - imagesize_offset.
      imagesize_offset: fixed int offset (default 32), NOT suggested by Optuna.
      layer_combos    : dict {key: [layer, ...]}; key is suggested categorically.
    """
    ss = cfg["search_space"]
    params: dict = {}

    # ── resize + imagesize ─────────────────────────────────────────────────────
    if "resize_imagesize_pairs" in ss:
        # Explicit pairs — needed for DINOv2 (imagesize must be a multiple of 14).
        pairs = ss["resize_imagesize_pairs"]
        pair_keys = ["{}x{}".format(r, i) for r, i in pairs]
        chosen = trial.suggest_categorical("resize_imagesize", pair_keys)
        idx = pair_keys.index(chosen)
        params["resize"]    = pairs[idx][0]
        params["imagesize"] = pairs[idx][1]
    else:
        resize_spec = ss["resize"]
        resize = trial.suggest_categorical("resize", resize_spec["choices"])
        offset = ss.get("imagesize_offset", 32)
        params["resize"]    = resize
        params["imagesize"] = resize - offset

    # ── layer_combos (categorical over keys) ───────────────────────────────────
    layer_combos = ss["layer_combos"]
    layer_key = trial.suggest_categorical("layer_key", list(layer_combos.keys()))
    params["layer_key"]             = layer_key
    params["layers_to_extract_from"] = layer_combos[layer_key]

    # ── all other search-space params (generic dispatch) ───────────────────────
    _skip = {"resize", "layer_combos", "imagesize_offset", "resize_imagesize_pairs"}
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
            raise ValueError(f"Unknown search_space type '{t}' for param '{name}'")

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
    coreset_pct = params["coreset_pct"]
    if coreset_pct >= 1.0:
        sampler = patchcore.sampler.IdentitySampler()
    else:
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
            percentage=coreset_pct, device=device
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

def _make_objective(cfg: dict, sweep_dir: str, device: torch.device):
    pilot_classes = cfg["pilot_classes"]

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, cfg)

        # Add fixed metadata needed by trial_name / YAML
        params["backbone"]      = cfg["backbone"]
        params["pilot_classes"] = pilot_classes
        params["seed"]          = cfg["seed"]

        trial_name = _trial_name(params, cfg)
        trial_dir  = os.path.join(sweep_dir, trial_name)
        os.makedirs(trial_dir, exist_ok=True)

        # Save config immediately (survives crashes)
        with open(os.path.join(trial_dir, "config.yaml"), "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)

        LOGGER.info(
            "[Trial %d] %s  coreset=%.4f",
            trial.number, trial_name, params["coreset_pct"],
        )

        # ── Run on each pilot class ───────────────────────────────────────────
        class_aurocs = {}
        for cls in pilot_classes:
            LOGGER.info("  -> %s", cls)
            try:
                auroc = _run_one_class(cls, params, cfg, device)
            except Exception as exc:
                LOGGER.error("  [FAILED] %s: %s", cls, exc, exc_info=True)
                raise optuna.TrialPruned(f"Class {cls} failed: {exc}")

            class_aurocs[cls] = auroc
            LOGGER.info("     AUROC = %.4f", auroc)

            # Intermediate reporting lets MedianPruner cut bad trials early
            trial.report(float(np.mean(list(class_aurocs.values()))), step=len(class_aurocs))
            if trial.should_prune():
                raise optuna.TrialPruned("Pruned by MedianPruner.")

        mean_auroc = float(np.mean(list(class_aurocs.values())))

        with open(os.path.join(trial_dir, "scores.yaml"), "w") as f:
            yaml.dump(
                {**class_aurocs, "mean_auroc": mean_auroc},
                f, default_flow_style=False, sort_keys=False,
            )

        LOGGER.info("[Trial %d] Mean AUROC = %.4f", trial.number, mean_auroc)
        return mean_auroc

    return objective

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

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
                "pilot_classes", "search_space", "seed", "n_trials",
                "batch_size", "num_workers"]
    for key in required:
        if key not in cfg:
            raise KeyError(f"Missing required key in sweep config: '{key}'")

    seed        = cfg["seed"]
    n_trials    = cfg["n_trials"]
    gpu_id      = cfg.get("gpu", 0)
    results_root = cfg.get("results_root", "results")

    dataset_prefix = _DATASET_PREFIX.get(cfg["dataset"], cfg["dataset"].upper())
    sweep_dir = os.path.join(results_root, f"{dataset_prefix}_Sweep_{study_name}")
    os.makedirs(sweep_dir, exist_ok=True)

    storage_url = f"sqlite:///{os.path.abspath(os.path.join(sweep_dir, study_name + '.db'))}"

    LOGGER.info("Study       : %s", study_name)
    LOGGER.info("Config      : %s", config or os.path.join(_CONFIGS_DIR, f"{study_name}.yaml"))
    LOGGER.info("Backbone    : %s  (%s)", cfg["backbone"], cfg["backbone_short"])
    LOGGER.info("Dataset     : %s  (%s)", cfg["dataset"], cfg["data_path"])
    LOGGER.info("Pilot cls   : %s", cfg["pilot_classes"])
    LOGGER.info("Layer combos: %s", list(cfg["search_space"]["layer_combos"].keys()))
    LOGGER.info("n_trials    : %d", n_trials)
    LOGGER.info("Sweep dir   : %s", os.path.abspath(sweep_dir))

    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu"
    )
    LOGGER.info("Device      : %s", device)

    # ── Optuna study ──────────────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

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

    objective = _make_objective(cfg, sweep_dir, device)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,),
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        LOGGER.warning("No trials completed successfully.")
        return

    best = study.best_trial
    LOGGER.info("=" * 60)
    LOGGER.info("Best trial  #%d", best.number)
    LOGGER.info("Mean AUROC  : %.4f", best.value)
    LOGGER.info("Params      : %s", best.params)

    # Reconstruct full params for the best trial name
    best_params = dict(best.params)
    best_params["imagesize"] = best_params["resize"] - cfg["search_space"].get("imagesize_offset", 32)
    best_params["layers_to_extract_from"] = cfg["search_space"]["layer_combos"][best_params["layer_key"]]
    # fill missing keys that _trial_name needs (sampled but not in best.params dict)
    for k in ("pretrain_embed_dimension", "target_embed_dimension",
               "patchsize", "coreset_pct", "anomaly_scorer_num_nn"):
        if k not in best_params:
            best_params[k] = best.params.get(k)

    summary = {
        "study_name":          study_name,
        "n_completed_trials":  len(completed),
        "best_trial_number":   best.number,
        "best_mean_auroc":     best.value,
        "best_params":         dict(best.params),
        "best_trial_name":     _trial_name(best_params, cfg),
    }
    summary_path = os.path.join(sweep_dir, "best_trial.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    LOGGER.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
