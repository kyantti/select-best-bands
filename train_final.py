"""Train the final model on every train crop, then read the test set once per seed.

    uv run python train_final.py N                  # the winner of experiment N
    uv run python train_final.py N --bands R G B    # a triplet named by hand
    uv run python train_final.py N --seeds 1 2 3    # one final model per seed

The order of this script is the claim it makes.  Every model is fitted on all
28 train Acquisitions and saved to disk before a single held-out crop is
opened, and the test crops are then predicted in one ordered pass per model.
Nothing that happens after the save can reach back into training, so the test
metrics are held-out measurements rather than numbers the run was tuned towards.

`config.FINAL_SEEDS` is a list because the noise of training and the noise of
the split are different things.  With its default single seed this is the 10
Sep run, file for file; with several it writes one model and one set of tables
per seed plus a summary of their mean and spread, and every artifact declares
how many times the held-out set was read.  That count is the length of the
list: ten seeds are ten reads, and no file of such a run may claim one.

The training loop, the inference pass and the frozen normalization are copied
from fig-aflatoxin's final stage; the experiment-numbered outputs are this
repo's.  Note that only `engine.train_step` is called while training: the
epoch loop has no evaluation half, because there is no loader to evaluate on.
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

import config
import ga
from cnn import engine
from cnn.data_setup import (
    Hypercube,
    SelectedBandDataset,
    fit_foreground_normalization,
    load_partition_crops,
    load_spectral_axis,
    wavelengths_of,
)
from cnn.model import (
    build_resnet50,
    checkpoint_identity,
    checkpoint_path,
    classification_metrics,
    predict,
    seed_everything,
)

matplotlib.use("Agg")  # the final run goes under nohup like the search
import matplotlib.pyplot as plt  # noqa: E402
import seaborn  # noqa: E402

Candidate = tuple[int, int, int]


# --- What the two halves of the run are handed ------------------------------


@dataclass(frozen=True)
class FinalTraining:
    """Everything the final training run sees.

    There is no test crop in it, and no field that could lead to one: the
    held-out manifest rows have not been read yet when this is built.
    """

    bands: Candidate
    training: list[Hypercube]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    seed: int
    device: torch.device
    epochs: int
    # The pretrained backbone this run starts from, or None for ImageNet alone.
    checkpoint: Path | None = None


@dataclass(frozen=True)
class FinalEvaluation:
    """The held-out crops, the frozen model and the normalization it was fitted under."""

    bands: Candidate
    test: list[Hypercube]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    state: dict[str, torch.Tensor]
    device: torch.device


@dataclass(frozen=True)
class FinalModel:
    """The trained weights, on the CPU, and one history row per epoch."""

    state: dict[str, torch.Tensor]
    history: list[dict]


# --- Training and the single inference pass ---------------------------------


def train_final_model(
    context: FinalTraining,
    *,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    size: tuple[int, int] = config.IMAGE_SIZE,
    num_workers: int = config.NUM_WORKERS,
    verbose: bool = True,
) -> FinalModel:
    """Fit the band triplet on every train crop, with no evaluation half.

    Copied from fig-aflatoxin's `train_final_resnet`: the generators are seeded
    first, the loader is built next and the model after it, so the weights are
    drawn from the same stream the recorded run drew them from.
    """
    generator = seed_everything(context.seed)
    train_loader = DataLoader(
        SelectedBandDataset(
            context.training, context.bands, context.mean, context.std, size, training=True
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=context.device.type == "cuda",
        generator=generator,
    )
    model = build_resnet50(context.device, context.checkpoint)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler(device="cuda") if context.device.type == "cuda" else None
    history = []
    for epoch in range(1, context.epochs + 1):
        # `train_step`, never `engine.train`: that one takes a second loader and
        # evaluates on it every epoch, which is exactly what must not happen here.
        train_loss, train_accuracy = engine.train_step(
            model, train_loader, loss_fn, optimizer, context.device, scaler
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy}
        )
        if verbose:
            print(
                f"  epoch {epoch:>3}/{context.epochs}  "
                f"train_loss {train_loss:.4f}  train_accuracy {train_accuracy:.4f}",
                flush=True,
            )
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return FinalModel(state, history)


def predict_test(
    context: FinalEvaluation,
    *,
    batch_size: int = config.BATCH_SIZE,
    size: tuple[int, int] = config.IMAGE_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> list[int]:
    """Load the frozen weights and predict the held-out crops once, in order.

    No checkpoint here even when the training run had one: every parameter of
    this model comes from the state dict that was just saved, so loading a
    backbone first would only be overwritten.
    """
    test_loader = DataLoader(
        SelectedBandDataset(
            context.test, context.bands, context.mean, context.std, size, training=False
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=context.device.type == "cuda",
    )
    model = build_resnet50(context.device)
    model.load_state_dict(context.state)
    return predict(model, test_loader, context.device)


# --- The data -------------------------------------------------------------


def winner_bands(experiment: int) -> Candidate:
    """The band triplet experiment `N`'s search chose, from its own summary.

    Raises:
        ValueError: If the summary is absent or does not name three bands.
    """
    path = ga.experiment_paths(experiment)["summary"]
    try:
        summary = json.loads(path.read_text())
        indices = tuple(int(index) for index in summary["winner"]["selected_band_indices"])
        if len(indices) != 3:
            raise ValueError("a winner is three band indices")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as unreadable:
        raise ValueError(
            f"cannot read the winner from {path.name} ({unreadable}); "
            f"run the search of experiment {experiment} first, or pass --bands R G B"
        ) from unreadable
    return indices


# --- The outputs ----------------------------------------------------------


def final_paths(experiment: int, bands: Candidate, seed: int | None = None) -> dict[str, Path]:
    """Every file one final model owns.  Another triplet writes to other names.

    `seed` is the seed of a run that trained several models, and names the
    files of that one: a list of seeds has to keep its models apart.  The
    single-seed run passes None and writes the names of the 10 Sep run, so an
    artifact of this protocol is never renamed by a feature it did not use.
    """
    prefix = f"{ga.experiment_prefix(experiment)}_"
    suffix = ga.band_suffix(bands) + ("" if seed is None else f"_seed{int(seed)}")
    return {
        "model": config.MODELS_DIR / f"{prefix}model_{suffix}.pt",
        "metrics": config.TABLES_DIR / f"{prefix}final_metrics_{suffix}.json",
        "confusion": config.TABLES_DIR / f"{prefix}confusion_matrix_{suffix}.csv",
        "predictions": config.TABLES_DIR / f"{prefix}test_predictions_{suffix}.csv",
        "history": config.TABLES_DIR / f"{prefix}training_history_{suffix}.csv",
        "confusion_figure": config.FIGURES_DIR / f"{prefix}confusion_matrix_{suffix}.png",
        "history_figure": config.FIGURES_DIR / f"{prefix}training_history_{suffix}.png",
    }


def final_summary_path(experiment: int, bands: Candidate) -> Path:
    """Where a run of several seeds says what the spread of its seeds was.

    It belongs to the triplet rather than to any one seed, and a single-seed
    run does not write it: with one model there is nothing to average.
    """
    prefix = ga.experiment_prefix(experiment)
    return config.TABLES_DIR / f"{prefix}_final_summary_{ga.band_suffix(bands)}.json"


def resolve_seeds(seeds=None) -> list[int]:
    """The training seeds of this run, checked before anything is trained.

    Raises:
        ValueError: If the list is empty or names a seed twice — two runs of
            one seed would write one model's files twice and count as two
            reads of the test set while measuring the same thing once.
    """
    resolved = [int(seed) for seed in (config.FINAL_SEEDS if seeds is None else seeds)]
    if not resolved:
        raise ValueError("a final run needs at least one training seed")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"the final seeds {resolved} repeat a seed; each one trains one model")
    return resolved


def refuse_a_number_that_already_holds_another_seeding(
    experiment: int, bands: Candidate, seeds: list[int]
) -> None:
    """Refuse to put a multi-seed result beside a single-seed one, or the reverse.

    Neither would overwrite the other — the names differ — so the settings
    guard never sees them.  But the number would then carry two answers about
    the same held-out crops, and the unsuffixed one would say it was read once
    while the `_seed` files said three.  One experiment number is one claim.

    Raises:
        ValueError: If this triplet already has a result of the other shape.
    """
    prefix = ga.experiment_prefix(experiment)
    suffix = ga.band_suffix(bands)
    single = config.TABLES_DIR / f"{prefix}_final_metrics_{suffix}.json"
    several = sorted(config.TABLES_DIR.glob(f"{prefix}_final_metrics_{suffix}_seed*.json"))
    if len(seeds) > 1 and single.exists():
        raise ValueError(
            f"{single.name} is a single-seed result of {prefix}, and {len(seeds)} seeds "
            "read the held-out set that many times. Two claims cannot share one "
            "experiment number. Use a new experiment number."
        )
    if len(seeds) == 1 and several:
        raise ValueError(
            f"{several[0].name} is part of a {len(several)}-seed result of {prefix}, and a "
            "single-seed run beside it would claim one read of the held-out set. "
            "Use a new experiment number."
        )


def refuse_a_number_an_earlier_protocol_owns(experiment: int) -> None:
    """Refuse a number whose `out/` files were written by a run this code is not.

    Experiments 1–20 own their `exp_NN_confusion_matrix_*.csv` and their
    figures, and those are names this script writes: without this check,
    `train_final.py 20` would quietly replace a thesis result.  A number is this
    protocol's once its search has left a candidate cache or a summary, or once
    this script has already trained a final model for it; a number with nothing
    under `out/` is free.

    Raises:
        ValueError: If the number is neither free nor this protocol's.
    """
    prefix = ga.experiment_prefix(experiment)
    ours = (
        config.TABLES_DIR / f"{prefix}_candidates.csv",
        config.TABLES_DIR / f"{prefix}_ga_summary.json",
    )
    if any(path.exists() for path in ours):
        return
    if list(config.TABLES_DIR.glob(f"{prefix}_final_metrics_*.json")):
        return
    occupied = [
        path
        for directory in (config.TABLES_DIR, config.FIGURES_DIR)
        for path in directory.glob(f"{prefix}_*")
    ]
    if occupied:
        raise ValueError(
            f"{occupied[0]} belongs to an earlier run that this script cannot reproduce, "
            f"so experiment {prefix} is not free. Use a new experiment number."
        )


# What a rerun has to match for it to be a replay; the rule itself is ga.py's,
# because the search, the final model and the bootstrap all obey the same one.
RECORDED_SETTINGS = (
    "selected_band_indices",
    "seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "image_size",
    "device_type",
    "backbone_checkpoint",
    # Not a setting that moves the number, but one that moves what the file
    # claims: rewriting a seed's result under a shorter list would turn three
    # reads of the held-out set into two.
    "test_evaluation_count",
)


def run_settings(
    bands: Candidate,
    device: torch.device,
    epochs: int,
    seed: int,
    read_count: int,
    checkpoint: Path | None = None,
) -> dict:
    """What a rerun has to match to be a replay rather than a different result.

    `backbone_checkpoint` is null when there is none, which is what a metrics
    file written before this ticket says by leaving the key out: a result
    recorded without a checkpoint can still be replayed without one.
    """
    return {
        "selected_band_indices": [int(band) for band in bands],
        "seed": seed,
        "epochs": epochs,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "image_size": list(config.IMAGE_SIZE),
        "device_type": device.type,
        "backbone_checkpoint": checkpoint_identity(checkpoint),
        "test_evaluation_count": read_count,
    }


def write_confusion_matrix(path: Path, matrix: list[list[int]]) -> None:
    """The confusion matrix with its classes named, as fig-aflatoxin writes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("actual", *(f"predicted_{name}" for name in config.CLASS_NAMES)))
        for name, row in zip(config.CLASS_NAMES, matrix, strict=True):
            writer.writerow((name, *row))


def write_history(path: Path, history: list[dict]) -> None:
    """One row per epoch: the loss and accuracy of the training crops alone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=("epoch", "train_loss", "train_accuracy"), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(history)


def plot_confusion_matrix(
    path: Path, matrix: list[list[int]], nanometres: tuple[float, float, float]
) -> None:
    """The confusion matrix as the thesis shows it, in counts, with nm in the title."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(6, 5))
    seaborn.heatmap(
        np.asarray(matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=list(config.CLASS_NAMES),
        yticklabels=list(config.CLASS_NAMES),
        ax=axes,
    )
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")
    axes.set_title(
        "Held-out test — " + ", ".join(f"{value:.2f} nm" for value in nanometres)
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_training_history(
    path: Path, history: list[dict], nanometres: tuple[float, float, float]
) -> None:
    """Loss and accuracy per epoch, side by side.  Neither curve is a test curve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, (loss_axes, accuracy_axes) = plt.subplots(1, 2, figsize=(11, 4.5))
    # Marked points, so a one-epoch smoke run draws something instead of an empty box.
    loss_axes.plot(epochs, [row["train_loss"] for row in history], color="tab:blue", marker="o", markersize=3)
    loss_axes.set_xlabel("Epoch")
    loss_axes.set_ylabel("Training loss")
    accuracy_axes.plot(
        epochs, [row["train_accuracy"] for row in history], color="tab:green", marker="o", markersize=3
    )
    accuracy_axes.set_xlabel("Epoch")
    accuracy_axes.set_ylabel("Training accuracy")
    for axes in (loss_axes, accuracy_axes):
        axes.grid(True, alpha=0.3)
    figure.suptitle(
        "Final training — " + ", ".join(f"{value:.2f} nm" for value in nanometres)
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_metrics(
    path: Path,
    experiment: int,
    context: FinalTraining,
    test: list[Hypercube],
    metrics: dict,
    nanometres: tuple[float, float, float],
    seeds: list[int],
) -> None:
    """The one file that says what this model is and what it scored, once.

    It carries no timestamp and no elapsed time on purpose: two runs of the
    same experiment number and triplet write the same bytes, so a rerun that
    changed something would show up as a diff.  `backbone_checkpoint` appears
    only when the model was given one, so a run without one still writes the
    file this protocol recorded.

    `test_evaluation_count` is the length of `seeds`, never one per file: ten
    seeds are ten trainings scored on the same held-out crops, and an artifact
    that said "1" would claim a single measurement that was taken ten times.
    `final_seeds` joins it only when there are several, for the same reason
    `backbone_checkpoint` does — a single-seed run writes the 10 Sep file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_identity(context.checkpoint)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                **({} if checkpoint is None else {"backbone_checkpoint": checkpoint}),
                "experiment": experiment,
                **({} if len(seeds) == 1 else {"final_seeds": list(seeds)}),
                "primary_metric": "weighted_f1",
                "evaluation_scope": "held-out-test-once-after-final-training",
                "test_evaluation_count": len(seeds),
                "test_feedback_used": False,
                "selected_band_indices": [int(band) for band in context.bands],
                "selected_wavelengths_nm": list(nanometres),
                "seed": context.seed,
                "epochs": context.epochs,
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "image_size": list(config.IMAGE_SIZE),
                "device_type": context.device.type,
                "normalization": {
                    "fitted_on": "train-foreground",
                    "mean": list(context.mean),
                    "std": list(context.std),
                },
                "train_cropped_hypercube_count": len(context.training),
                "train_acquisition_count": len(
                    {crop.acquisition_id for crop in context.training}
                ),
                "test_cropped_hypercube_count": len(test),
                "test_acquisition_count": len({crop.acquisition_id for crop in test}),
                "metrics": metrics,
                "libraries": ga.library_versions(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


# --- What a list of seeds measured -----------------------------------------


# The four scalars of `classification_metrics`; per-class recall is a table and
# is left to the per-seed files rather than averaged into a single number.
SUMMARY_METRICS = ("weighted_f1", "macro_f1", "ordinal_mae", "quadratic_weighted_kappa")

# The identity of a summary: the triplet, what every seed was trained under,
# and which seeds they were.  A list in another order measured another thing.
SUMMARY_RECORDED_SETTINGS = (
    "selected_band_indices",
    "final_seeds",
    "epochs",
    "batch_size",
    "learning_rate",
    "image_size",
    "device_type",
    "backbone_checkpoint",
    "test_evaluation_count",
)


def summary_settings(
    bands: Candidate,
    device: torch.device,
    epochs: int,
    seeds: list[int],
    checkpoint: Path | None = None,
) -> dict:
    """What a rerun of the whole list has to match for the summary to be a replay.

    It is the per-seed contract with the one seed taken out and the whole list
    put in: a summary belongs to the list, and a list of another length or in
    another order summarizes something else.
    """
    settings = run_settings(bands, device, epochs, seeds[0], len(seeds), checkpoint)
    settings.pop("seed")
    return {**settings, "final_seeds": list(seeds)}


def spread_over_seeds(values: list[float]) -> dict:
    """Where the seeds landed on one metric: the mean and how far they sat from it.

    The standard deviation is the sample one, over at least two seeds — a
    summary is only written when there are several models to compare.  It is
    spelled out rather than taken from `statistics.stdev`, which raises on an
    undefined metric instead of carrying the NaN through: a kappa that has no
    value must leave the other three metrics' summaries standing.

    One undefined seed makes the whole summary of that metric undefined, min
    and max included.  `min` and `max` would otherwise answer by where the NaN
    happened to sit in the list, quoting two finite seeds as if all of them had
    scored — and the same seeds in another order would write other bytes.
    """
    if any(math.isnan(value) for value in values):
        return dict.fromkeys(("mean", "std", "min", "max"), math.nan)
    mean = statistics.fmean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {"mean": mean, "std": math.sqrt(variance), "min": min(values), "max": max(values)}


def write_final_summary(
    path: Path,
    experiment: int,
    bands: Candidate,
    nanometres: tuple[float, float, float],
    scored: dict[int, dict],
    device: torch.device,
    epochs: int,
    checkpoint: Path | None,
) -> None:
    """The one file that separates the noise of training from the noise of the split.

    It is the headline of a multi-seed run and says, in the same breath, that
    the held-out set was read once per seed.  Like the per-seed metrics it
    carries no timestamp, so a replay of the same list writes the same bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    seeds = list(scored)
    identity = checkpoint_identity(checkpoint)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                **({} if identity is None else {"backbone_checkpoint": identity}),
                "experiment": experiment,
                "primary_metric": "weighted_f1",
                "evaluation_scope": "held-out-test-once-per-seed-after-final-training",
                "test_evaluation_count": len(seeds),
                "test_feedback_used": False,
                "final_seeds": seeds,
                "selected_band_indices": [int(band) for band in bands],
                "selected_wavelengths_nm": list(nanometres),
                "epochs": epochs,
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "image_size": list(config.IMAGE_SIZE),
                "device_type": device.type,
                "per_seed": {
                    str(seed): {name: scored[seed][name] for name in SUMMARY_METRICS}
                    for seed in seeds
                },
                "metrics": {
                    name: spread_over_seeds([scored[seed][name] for seed in seeds])
                    for name in SUMMARY_METRICS
                },
                "libraries": ga.library_versions(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


# --- One final model ------------------------------------------------------


def final_command(
    experiment: int,
    *,
    bands=None,
    epochs: int = config.NUM_EPOCHS,
    checkpoint: Path | None = None,
    seeds=None,
    trainer=train_final_model,
    evaluator=predict_test,
    verbose: bool = True,
) -> None:
    """Train the final model of one experiment number, once per seed, and score it.

    `checkpoint` and `seeds` are resolved here rather than taken as given, so
    that `config.BACKBONE_CHECKPOINT` and `config.FINAL_SEEDS` apply to every
    caller and not only to the command line.

    With one seed this is the 10 Sep run and writes that run's files.  With
    several, every seed trains its own final model on the same train crops and
    the same frozen normalization, and every artifact declares that the
    held-out set was read once per seed.  All the models are trained and saved
    before the test partition is opened, so ticket 07's order holds for the
    last seed as strictly as for the first.

    Raises:
        ValueError: If a checkpoint was asked for and is not there, if the seed
            list is empty or repeats a seed, if the number belongs to an
            earlier protocol, if the bands are not a triplet of this
            SpectralAxis, or if the trainer returns something that is not a
            model with one history row per epoch.
    """
    checkpoint = checkpoint_path(checkpoint)
    refuse_a_number_an_earlier_protocol_owns(experiment)
    seeds = resolve_seeds(seeds)
    device = ga.select_device()
    # Resolved before a single NPZ is opened: a missing summary or a band out of
    # range is a mistake worth hearing about now, not in three minutes.
    selected = tuple(int(band) for band in bands) if bands is not None else winner_bands(experiment)
    wavelengths = load_spectral_axis(config.SPECTRAL_AXES)
    nanometres = wavelengths_of(selected, wavelengths)
    # One seed keeps the unsuffixed names of the recorded run; several are told
    # apart by the seed that trained them.
    paths_of = {
        seed: final_paths(experiment, selected, seed if len(seeds) > 1 else None) for seed in seeds
    }
    summary_path = final_summary_path(experiment, selected) if len(seeds) > 1 else None
    refuse_a_number_that_already_holds_another_seeding(experiment, selected, seeds)
    # Every guard before any training: a refusal on the last seed must not leave
    # the first seed's model already trained, saved and half-recorded.
    for seed in seeds:
        ga.refuse_a_rerun_that_would_not_reproduce(
            paths_of[seed]["metrics"],
            run_settings(selected, device, epochs, seed, len(seeds), checkpoint),
            RECORDED_SETTINGS,
        )
    if summary_path is not None:
        ga.refuse_a_rerun_that_would_not_reproduce(
            summary_path,
            summary_settings(selected, device, epochs, seeds, checkpoint),
            SUMMARY_RECORDED_SETTINGS,
        )

    training = load_partition_crops(
        config.HYPERCUBES_MANIFEST, "train", wavelengths, verbose=verbose
    )
    # Fitted once and shared: the normalization depends on the train crops and
    # the triplet, never on the seed, so every seed trains under the same one.
    mean, std = fit_foreground_normalization(training, selected)
    contexts = {
        seed: FinalTraining(selected, training, mean, std, seed, device, epochs, checkpoint)
        for seed in seeds
    }
    if verbose:
        print(f"candidate      {selected}")
        print("wavelengths_nm " + ", ".join(f"{value}" for value in nanometres))
        print(f"train crops    {len(training)} in {len({c.acquisition_id for c in training})} acquisitions")
        print(f"mean           {list(mean)}")
        print(f"std            {list(std)}")
        seeding = f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {seeds}"
        print(f"device         {device}  |  epochs {epochs}  |  {seeding}")
        if checkpoint is not None:
            ga.print_checkpoint(checkpoint)

    # Every model is held until the last one is trained, so that no held-out
    # crop is opened while a seed is still training.  A ResNet50 state dict is
    # about 100 MB on the CPU, so even ten seeds cost a gigabyte of RAM.
    models = {}
    for seed in seeds:
        started = time.perf_counter()
        model = trainer(contexts[seed])
        if not model.state or len(model.history) != epochs:
            raise ValueError(
                f"the final trainer returned no weights for seed {seed}, "
                "or not one history row per epoch"
            )
        paths_of[seed]["model"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state, paths_of[seed]["model"])
        models[seed] = model
        if verbose:
            # Flushed: under nohup these lines are the record of when each model
            # was frozen and when the held-out crops were first opened, and a
            # line still sitting in a buffer says nothing about that order.
            print(
                f"model          {paths_of[seed]['model']} "
                f"({time.perf_counter() - started:.1f} s)",
                flush=True,
            )
    if verbose:
        print("--- the test set is opened for the first time, after the save ---", flush=True)

    test = load_partition_crops(config.HYPERCUBES_MANIFEST, "test", wavelengths, verbose=verbose)
    scored = {}
    for seed in seeds:
        paths = paths_of[seed]
        model = models[seed]
        predictions = evaluator(
            FinalEvaluation(selected, test, mean, std, model.state, device)
        )
        if len(predictions) != len(test) or any(
            type(value) is not int or value not in range(config.NUM_CLASSES)
            for value in predictions
        ):
            raise ValueError(f"the final evaluator returned invalid predictions for seed {seed}")
        metrics = classification_metrics(
            [int(crop.severity_class) for crop in test], predictions
        )
        scored[seed] = metrics

        # ga.py owns the predictions schema; ticket 08's bootstrap reads both files.
        ga.write_predictions(paths["predictions"], test, predictions, exclusive=False)
        write_confusion_matrix(paths["confusion"], metrics["confusion_matrix"])
        write_history(paths["history"], model.history)
        write_metrics(paths["metrics"], experiment, contexts[seed], test, metrics, nanometres, seeds)
        plot_confusion_matrix(paths["confusion_figure"], metrics["confusion_matrix"], nanometres)
        plot_training_history(paths["history_figure"], model.history, nanometres)

        if verbose:
            if len(seeds) > 1:
                print(f"--- seed {seed} ---")
            for name in SUMMARY_METRICS:
                print(f"{name:<14} {metrics[name]!r}")
            for name, value in metrics["per_class_recall"].items():
                print(f"recall {name:<7} {value!r}")
            print(f"confusion      {metrics['confusion_matrix']}")
            print(f"test crops     {len(test)} in {len({c.acquisition_id for c in test})} acquisitions")
            for name in ("metrics", "confusion", "predictions", "history"):
                print(f"{name:<14} {paths[name]}")
            for name in ("confusion_figure", "history_figure"):
                print(f"{name:<14} {paths[name]}")

    if summary_path is not None:
        write_final_summary(
            summary_path, experiment, selected, nanometres, scored, device, epochs, checkpoint
        )
        if verbose:
            print(f"--- {len(seeds)} seeds, {len(seeds)} reads of the held-out set ---")
            for name in SUMMARY_METRICS:
                across = spread_over_seeds([scored[seed][name] for seed in seeds])
                print(
                    f"{name:<14} mean {across['mean']!r}  std {across['std']!r}  "
                    f"min {across['min']!r}  max {across['max']!r}"
                )
            print(f"summary        {summary_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", type=int, help="the experiment number this model belongs to")
    parser.add_argument(
        "--bands",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        help="the triplet to train; the default is the winner of the experiment's search",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.NUM_EPOCHS, help="epochs to train for (smoke use)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="a pretrained backbone to fine-tune from; the default is"
        " config.BACKBONE_CHECKPOINT, and it is recorded beside the result",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        metavar="SEED",
        help="the training seeds, one final model and one read of the held-out set"
        " each; the default is config.FINAL_SEEDS",
    )
    arguments = parser.parse_args(argv)
    try:
        final_command(
            arguments.experiment,
            bands=arguments.bands,
            epochs=arguments.epochs,
            checkpoint=arguments.checkpoint,
            seeds=arguments.seeds,
        )
    except ValueError as refusal:
        print(f"train_final.py: {refusal}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
