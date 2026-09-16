"""Train the final model on every train crop, then read the test set once.

    uv run python train_final.py N                  # the winner of experiment N
    uv run python train_final.py N --bands R G B    # a triplet named by hand

The order of this script is the claim it makes.  The model is fitted on all 28
train Acquisitions and saved to disk before a single held-out crop is opened,
and the test crops are then predicted in one ordered pass.  Nothing that
happens after the save can reach back into training, so the test metrics are a
single held-out measurement rather than a number the run was tuned towards.

The training loop, the inference pass and the frozen normalization are copied
from fig-aflatoxin's final stage; the experiment-numbered outputs are this
repo's.  Note that only `engine.train_step` is called while training: the
epoch loop has no evaluation half, because there is no loader to evaluate on.
"""

import argparse
import csv
import json
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


def final_paths(experiment: int, bands: Candidate) -> dict[str, Path]:
    """Every file one final model owns.  Another triplet writes to other names."""
    prefix = f"{ga.experiment_prefix(experiment)}_"
    suffix = ga.band_suffix(bands)
    return {
        "model": config.MODELS_DIR / f"{prefix}model_{suffix}.pt",
        "metrics": config.TABLES_DIR / f"{prefix}final_metrics_{suffix}.json",
        "confusion": config.TABLES_DIR / f"{prefix}confusion_matrix_{suffix}.csv",
        "predictions": config.TABLES_DIR / f"{prefix}test_predictions_{suffix}.csv",
        "history": config.TABLES_DIR / f"{prefix}training_history_{suffix}.csv",
        "confusion_figure": config.FIGURES_DIR / f"{prefix}confusion_matrix_{suffix}.png",
        "history_figure": config.FIGURES_DIR / f"{prefix}training_history_{suffix}.png",
    }


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
)


def run_settings(
    bands: Candidate, device: torch.device, epochs: int, checkpoint: Path | None = None
) -> dict:
    """What a rerun has to match to be a replay rather than a different result.

    `backbone_checkpoint` is null when there is none, which is what a metrics
    file written before this ticket says by leaving the key out: a result
    recorded without a checkpoint can still be replayed without one.
    """
    return {
        "selected_band_indices": [int(band) for band in bands],
        "seed": config.FINAL_SEED,
        "epochs": epochs,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "image_size": list(config.IMAGE_SIZE),
        "device_type": device.type,
        "backbone_checkpoint": checkpoint_identity(checkpoint),
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
) -> None:
    """The one file that says what this model is and what it scored, once.

    It carries no timestamp and no elapsed time on purpose: two runs of the
    same experiment number and triplet write the same bytes, so a rerun that
    changed something would show up as a diff.  `backbone_checkpoint` appears
    only when the model was given one, so a run without one still writes the
    file this protocol recorded.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_identity(context.checkpoint)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                **({} if checkpoint is None else {"backbone_checkpoint": checkpoint}),
                "experiment": experiment,
                "primary_metric": "weighted_f1",
                "evaluation_scope": "held-out-test-once-after-final-training",
                "test_evaluation_count": 1,
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


# --- One final model ------------------------------------------------------


def final_command(
    experiment: int,
    *,
    bands=None,
    epochs: int = config.NUM_EPOCHS,
    checkpoint: Path | None = None,
    trainer=train_final_model,
    evaluator=predict_test,
    verbose: bool = True,
) -> None:
    """Train the final model of one experiment number and score it once.

    `checkpoint` is resolved here rather than taken as given, so that
    `config.BACKBONE_CHECKPOINT` applies to every caller and not only to the
    command line.

    Raises:
        ValueError: If a checkpoint was asked for and is not there, if the
            number belongs to an earlier protocol, if the bands are not a
            triplet of this SpectralAxis, or if the trainer returns something
            that is not a model with one history row per epoch.
    """
    checkpoint = checkpoint_path(checkpoint)
    refuse_a_number_an_earlier_protocol_owns(experiment)
    device = ga.select_device()
    # Resolved before a single NPZ is opened: a missing summary or a band out of
    # range is a mistake worth hearing about now, not in three minutes.
    selected = tuple(int(band) for band in bands) if bands is not None else winner_bands(experiment)
    wavelengths = load_spectral_axis(config.SPECTRAL_AXES)
    nanometres = wavelengths_of(selected, wavelengths)
    paths = final_paths(experiment, selected)
    settings = run_settings(selected, device, epochs, checkpoint)
    ga.refuse_a_rerun_that_would_not_reproduce(paths["metrics"], settings, RECORDED_SETTINGS)

    training = load_partition_crops(
        config.HYPERCUBES_MANIFEST, "train", wavelengths, verbose=verbose
    )
    mean, std = fit_foreground_normalization(training, selected)
    context = FinalTraining(
        selected, training, mean, std, config.FINAL_SEED, device, epochs, checkpoint
    )
    if verbose:
        print(f"candidate      {selected}")
        print("wavelengths_nm " + ", ".join(f"{value}" for value in nanometres))
        print(f"train crops    {len(training)} in {len({c.acquisition_id for c in training})} acquisitions")
        print(f"mean           {list(mean)}")
        print(f"std            {list(std)}")
        print(f"device         {device}  |  epochs {epochs}  |  seed {context.seed}")
        if checkpoint is not None:
            ga.print_checkpoint(checkpoint)

    started = time.perf_counter()
    model = trainer(context)
    if not model.state or len(model.history) != epochs:
        raise ValueError("the final trainer returned no weights, or not one history row per epoch")
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state, paths["model"])
    trained_in = time.perf_counter() - started
    if verbose:
        # Flushed: under nohup these two lines are the record of when the model
        # was frozen and when the held-out crops were first opened, and a line
        # still sitting in a buffer says nothing about the order it happened in.
        print(f"model          {paths['model']} ({trained_in:.1f} s)", flush=True)
        print("--- the test set is opened for the first time, after the save ---", flush=True)

    test = load_partition_crops(config.HYPERCUBES_MANIFEST, "test", wavelengths, verbose=verbose)
    predictions = evaluator(
        FinalEvaluation(selected, test, mean, std, model.state, device)
    )
    if len(predictions) != len(test) or any(
        type(value) is not int or value not in range(config.NUM_CLASSES) for value in predictions
    ):
        raise ValueError("the final evaluator returned invalid predictions")
    metrics = classification_metrics(
        [int(crop.severity_class) for crop in test], predictions
    )

    # ga.py owns the predictions schema; ticket 08's bootstrap reads both files.
    ga.write_predictions(paths["predictions"], test, predictions, exclusive=False)
    write_confusion_matrix(paths["confusion"], metrics["confusion_matrix"])
    write_history(paths["history"], model.history)
    write_metrics(paths["metrics"], experiment, context, test, metrics, nanometres)
    plot_confusion_matrix(paths["confusion_figure"], metrics["confusion_matrix"], nanometres)
    plot_training_history(paths["history_figure"], model.history, nanometres)

    if verbose:
        for name in ("weighted_f1", "macro_f1", "ordinal_mae", "quadratic_weighted_kappa"):
            print(f"{name:<14} {metrics[name]!r}")
        for name, value in metrics["per_class_recall"].items():
            print(f"recall {name:<7} {value!r}")
        print(f"confusion      {metrics['confusion_matrix']}")
        print(f"test crops     {len(test)} in {len({c.acquisition_id for c in test})} acquisitions")
        for name in ("metrics", "confusion", "predictions", "history"):
            print(f"{name:<14} {paths[name]}")
        for name in ("confusion_figure", "history_figure"):
            print(f"{name:<14} {paths[name]}")


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
    arguments = parser.parse_args(argv)
    try:
        final_command(
            arguments.experiment,
            bands=arguments.bands,
            epochs=arguments.epochs,
            checkpoint=arguments.checkpoint,
        )
    except ValueError as refusal:
        print(f"train_final.py: {refusal}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
