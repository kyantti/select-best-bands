"""Genetic search for the best band triplet, and the evaluation of a single one.

    uv run python ga.py --evaluate 366 262 225    # train and score one triplet
    uv run python ga.py N                         # the search of experiment N

The search itself arrives with ticket `protocolo-80-20/05`; what is here is the
path every candidate goes through, exposed on its own so it can be checked
against a recorded run before a search is built on top of it.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

import config
from cnn.data_setup import (
    Hypercube,
    load_hypercubes,
    load_partitions,
    load_spectral_axis,
    spectral_axis_id,
    wavelengths_of,
)
from cnn.model import candidate_seed, data_identity, evaluate_candidate


def select_device() -> torch.device:
    """Resolve `config.DEVICE`, refusing to quietly fall back.

    `config.DEVICE` is part of what makes a number reproducible, so an
    accelerator that was asked for and is not there is an error: falling back to
    the CPU would keep running and produce a figure nobody could reproduce.

    Raises:
        ValueError: If the configured device is not available.
    """
    if config.DEVICE == "cpu":
        return torch.device("cpu")
    if config.DEVICE.startswith("cuda") and torch.cuda.is_available():
        return torch.device(config.DEVICE)
    raise ValueError(f"config.DEVICE is '{config.DEVICE}', which this machine does not have")


def load_selection_data(*, verbose: bool = True):
    """Load the crops band selection is allowed to see, and their identity.

    Held-out test rows are dropped before a single NPZ is opened, so the search
    cannot read what it will finally be scored on.  Returns the train-fit and
    validation crops, the spectral axis, and the checksums of all three.

    Raises:
        ValueError: If the manifests leak across a boundary, if an artifact is
            malformed, or if the crops were cut against another SpectralAxis.
    """
    rows = load_partitions(config.PARTITIONS)
    train_rows = [row for row in rows if row["partition"] == "train"]
    fit_rows = [row for row in train_rows if row["selection_partition"] == "train"]
    validation_rows = [row for row in train_rows if row["selection_partition"] == "validation"]
    if verbose:
        print(f"loading {len(fit_rows)} train-fit and {len(validation_rows)} validation crops")
    training = load_hypercubes(config.HYPERCUBES_MANIFEST, fit_rows, verbose=verbose)
    validation = load_hypercubes(config.HYPERCUBES_MANIFEST, validation_rows, verbose=verbose)
    wavelengths = load_spectral_axis(config.SPECTRAL_AXES)
    axis_ids = {hypercube.spectral_axis_id for hypercube in (*training, *validation)}
    if axis_ids != {spectral_axis_id(wavelengths)}:
        raise ValueError("train crops were cut against another SpectralAxis")
    identity = data_identity(training, validation, train_rows, wavelengths)
    return training, validation, wavelengths, identity


def write_predictions(path: Path, validation: list[Hypercube], predictions: list[int]) -> None:
    """Write one row per validation crop, in the order the model saw them.

    Opened exclusively, as fig-aflatoxin opens it: a file under `out/` is part
    of the thesis record, so a second run of the same triplet has to be told to
    write elsewhere rather than quietly replace the first one's numbers.

    Raises:
        FileExistsError: If `path` is already there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("cropped_hypercube_id", "acquisition_id", "actual", "predicted"),
            lineterminator="\n",
        )
        writer.writeheader()
        for hypercube, prediction in zip(validation, predictions, strict=True):
            writer.writerow(
                {
                    "cropped_hypercube_id": hypercube.cropped_hypercube_id,
                    "acquisition_id": hypercube.acquisition_id,
                    "actual": config.CLASS_NAMES[hypercube.severity_class],
                    "predicted": config.CLASS_NAMES[prediction],
                }
            )


def evaluate_command(bands, *, epochs: int, predictions_path: Path | None) -> None:
    """Train one triplet on the train-fit crops and report its validation metrics.

    Raises:
        FileExistsError: If the predictions file is already there.  Checked
            before anything is loaded, so an hour of training is never spent on
            a result that cannot be written down.
    """
    if predictions_path is None:
        r, g, b = (int(band) for band in bands)
        predictions_path = config.TABLES_DIR / f"evaluate_{r}_{g}_{b}_validation_predictions.csv"
    if predictions_path.exists():
        raise FileExistsError(
            f"{predictions_path} is already there; results under out/ are not overwritten. "
            "Move it aside, or pass --predictions with another path."
        )
    device = select_device()
    training, validation, wavelengths, identity = load_selection_data()
    nanometres = wavelengths_of(bands, wavelengths)
    derived_seed = candidate_seed(bands, identity)
    print(f"candidate      {tuple(bands)}")
    print("wavelengths_nm " + ", ".join(f"{value}" for value in nanometres))
    print(f"candidate seed {derived_seed}")
    print(f"device         {device}  |  epochs {epochs}")

    started = time.perf_counter()
    predictions, metrics, (mean, std) = evaluate_candidate(
        bands,
        training,
        validation,
        seed=derived_seed,
        device=device,
        epochs=epochs,
    )
    elapsed = time.perf_counter() - started

    print(f"mean           {list(mean)}")
    print(f"std            {list(std)}")
    for name in ("weighted_f1", "macro_f1", "ordinal_mae", "quadratic_weighted_kappa"):
        print(f"{name:<14} {metrics[name]!r}")
    print(f"elapsed_s      {elapsed:.1f}")

    write_predictions(predictions_path, validation, predictions)
    print(f"predictions    {predictions_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # The experiment number is accepted, and then refused below, so that the
    # documented `ga.py N` says where the search went instead of dying on an
    # unrecognized argument.
    parser.add_argument(
        "experiment", nargs="?", type=int, help="experiment number of a search run"
    )
    parser.add_argument(
        "--evaluate",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        help="train and score one band triplet instead of searching",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.NUM_EPOCHS, help="epochs to train for (smoke use)"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="where to write the validation predictions; the default is under out/tables/"
        " and is never overwritten",
    )
    arguments = parser.parse_args(argv)

    if arguments.evaluate is None:
        parser.error("the genetic search arrives with ticket 05; use --evaluate R G B for now")
    if arguments.experiment is not None:
        parser.error("--evaluate takes no experiment number: it writes no exp_NN_ output")
    evaluate_command(
        tuple(arguments.evaluate),
        epochs=arguments.epochs,
        predictions_path=arguments.predictions,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
