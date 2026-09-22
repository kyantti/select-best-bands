"""Quote the held-out test result with its width, by resampling whole captures.

    uv run python bootstrap.py N                  # the final model of experiment N
    uv run python bootstrap.py N --bands R G B    # when N has more than one

No model runs here.  The script reads the per-crop predictions
`train_final.py` already wrote and resamples them, so the interval costs
seconds and can be recomputed at any time from what is under `out/`.

Whole Acquisitions are resampled, never single crops: the crops of one capture
share a fig, an exposure and a session, so treating them as independent draws
would report an interval far narrower than the evidence supports.  With eight
test captures the interval is wide, and that width is the point.

Copied from fig-aflatoxin's `scripts/bootstrap-test-interval.py`; the
experiment-numbered discovery and outputs are this repo's.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import config
import ga
import train_final

Candidate = tuple[int, int, int]

# The interval is arithmetic over a CSV, so these two are the only versions that
# can move it.  (`ga` and `train_final` are imported for the names every result
# file shares, which is what pulls torch in; nothing here uses it.)
LIBRARIES = ("numpy", "scikit-learn")


# --- The predictions one final model left behind ---------------------------


@dataclass(frozen=True)
class TestPredictions:
    """The held-out crops of one final model, as three aligned columns.

    Class names, not indices, exactly as they sit in the CSV: this file is read
    back long after the model is gone, and `C2` survives a reader that has
    never seen `config.CLASS_NAMES`.
    """

    actual: np.ndarray
    predicted: np.ndarray
    acquisition: np.ndarray

    def __len__(self) -> int:
        return len(self.actual)

    @property
    def captures(self) -> list[str]:
        """The test Acquisitions, in a fixed order, each named once."""
        return sorted(set(self.acquisition.tolist()))

    def within(self, index: np.ndarray) -> "TestPredictions":
        """The same predictions restricted to `index` — one capture, or one resample."""
        return TestPredictions(self.actual[index], self.predicted[index], self.acquisition[index])


# --- Scoring --------------------------------------------------------------
# The four class names are always passed as `labels`, the way
# `cnn.model.classification_metrics` passes the four indices: it keeps a class
# that no crop was predicted into in the macro average instead of dropping it,
# and it keeps the scorer quiet about zero division.  It does not change
# weighted F1, where an absent class carries zero weight either way — which is
# why the interval below is the same number the ported script produced.


def weighted_f1(predictions: TestPredictions) -> float:
    """Weighted F1, the same primary metric the search and the final model use."""
    return float(
        f1_score(
            predictions.actual,
            predictions.predicted,
            labels=list(config.CLASS_NAMES),
            average="weighted",
            zero_division=0,
        )
    )


def macro_f1(predictions: TestPredictions) -> float:
    """Macro F1, reported beside it because every earlier experiment reports it."""
    return float(
        f1_score(
            predictions.actual,
            predictions.predicted,
            labels=list(config.CLASS_NAMES),
            average="macro",
            zero_division=0,
        )
    )


def accuracy(predictions: TestPredictions) -> float:
    """The plain hit rate, which the per-capture table quotes beside weighted F1."""
    return float(accuracy_score(predictions.actual, predictions.predicted))


# --- The grouped resample -------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A point estimate, the percentile interval around it, and what made it."""

    point: float
    low: float
    high: float
    scores: np.ndarray


def grouped_resamples(acquisition: np.ndarray, *, resamples: int, seed: int) -> list[np.ndarray]:
    """Draw `resamples` bootstrap samples of whole Acquisitions, with replacement.

    Each draw picks as many captures as the test set has and takes every crop
    of each one, so a capture is in or out as a block and can appear twice.
    The generator is created here and used in one order, which is what makes a
    quoted interval reproducible.
    """
    captures = sorted(set(acquisition.tolist()))
    members = {capture: np.flatnonzero(acquisition == capture) for capture in captures}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        picked = rng.choice(len(captures), size=len(captures), replace=True)
        draws.append(np.concatenate([members[captures[index]] for index in picked]))
    return draws


def bootstrap_interval(predictions: TestPredictions, *, resamples: int, seed: int) -> Interval:
    """The weighted F1 of the whole test set, and its percentile interval."""
    scores = np.array(
        [
            weighted_f1(predictions.within(index))
            for index in grouped_resamples(
                predictions.acquisition, resamples=resamples, seed=seed
            )
        ]
    )
    low, high = np.percentile(scores, list(config.BOOTSTRAP_PERCENTILES))
    return Interval(weighted_f1(predictions), float(low), float(high), scores)


def per_acquisition_scores(predictions: TestPredictions) -> list[dict]:
    """One row per test capture, worst first.

    The interval's width comes from somewhere, and this table says where: it is
    what tells a weak capture apart from a weak class.
    """
    table = []
    for capture in predictions.captures:
        index = np.flatnonzero(predictions.acquisition == capture)
        inside = predictions.within(index)
        labels, counts = np.unique(inside.actual, return_counts=True)
        table.append(
            {
                "acquisition_id": capture,
                "majority_class": str(labels[np.argmax(counts)]),
                "cropped_hypercube_count": len(inside),
                "weighted_f1": weighted_f1(inside),
                "accuracy": accuracy(inside),
            }
        )
    # The id breaks ties, so two equally weak captures keep a stable order.
    table.sort(key=lambda row: (row["weighted_f1"], row["acquisition_id"]))
    return table


# --- Finding the final model this experiment published ---------------------


@dataclass(frozen=True)
class FinalResult:
    """The published final model an interval is computed for, and its predictions."""

    experiment: int
    bands: Candidate
    nanometres: list[float]
    source: Path  # the predictions file the numbers were read from
    predictions: TestPredictions


def bootstrap_paths(experiment: int, bands: Candidate) -> dict[str, Path]:
    """The two files read and the one written, for one experiment and triplet."""
    paths = train_final.final_paths(experiment, bands)
    return {
        "metrics": paths["metrics"],
        "predictions": paths["predictions"],
        "bootstrap": config.TABLES_DIR
        / f"{ga.experiment_prefix(experiment)}_bootstrap_{ga.band_suffix(bands)}.json",
    }


def published_triplets(experiment: int) -> list[Candidate]:
    """Every band triplet experiment `N` has a final model for, from its metrics files.

    Each triplet appears once however many files name it: a multi-seed run of
    `train_final.py` writes one metrics file per seed, and three files of one
    triplet are one triplet, not three to choose between.
    """
    prefix = ga.experiment_prefix(experiment)
    found = []
    for path in sorted(config.TABLES_DIR.glob(f"{prefix}_final_metrics_*.json")):
        try:
            recorded = json.loads(path.read_text())["selected_band_indices"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as unreadable:
            raise ValueError(f"cannot read the bands from {path.name}: {unreadable}") from unreadable
        triplet = tuple(int(band) for band in recorded)
        if triplet not in found:
            found.append(triplet)
    return found


def final_bands(experiment: int, bands=None) -> Candidate:
    """Which triplet to bootstrap: the experiment's only one, or the one asked for.

    Raises:
        ValueError: If the experiment has no final model, if it has several and
            none was named, or if the one named is not among them.
    """
    published = published_triplets(experiment)
    prefix = ga.experiment_prefix(experiment)
    if bands is not None:
        selected = tuple(int(band) for band in bands)
        if selected not in published:
            raise ValueError(
                f"{prefix} has no final model for {selected}; "
                f"it has {published or 'none'}. Run train_final.py first."
            )
        return selected
    if not published:
        raise ValueError(
            f"{prefix} has no final model under {config.TABLES_DIR}, so there are no "
            f"test predictions to resample. Run train_final.py {experiment} first."
        )
    if len(published) > 1:
        raise ValueError(
            f"{prefix} has final models for {published}; say which one with --bands R G B."
        )
    return published[0]


def read_predictions(path: Path) -> TestPredictions:
    """The actual class, the predicted class and the capture of every test crop.

    Raises:
        ValueError: If the file is not there, or is not a predictions file.
    """
    if not path.exists():
        raise ValueError(
            f"{path.name} is not under {path.parent}; run train_final.py for this triplet first."
        )
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    columns = ("actual", "predicted", "acquisition_id")
    if not rows or any(column not in rows[0] for column in columns):
        raise ValueError(f"{path.name} does not have the columns {columns} of a predictions file")
    return TestPredictions(*(np.array([row[column] for row in rows]) for column in columns))


def refuse_a_multi_seed_result(experiment: int, bands: Candidate, predictions: Path) -> None:
    """Say so plainly when the triplet was trained under several seeds.

    This script resamples one set of per-crop predictions, and a run of several
    seeds wrote one set per seed.  Which interval such a run has — one per
    seed, or one over the pooled predictions — is a protocol decision nobody
    has taken yet, so refuse rather than guess, and refuse with the names of
    the files that are actually there.

    Raises:
        ValueError: If the unsuffixed predictions are absent and per-seed ones exist.
    """
    if predictions.exists():
        return
    prefix = ga.experiment_prefix(experiment)
    per_seed = sorted(config.TABLES_DIR.glob(f"{prefix}_test_predictions_{ga.band_suffix(bands)}_seed*.csv"))
    if per_seed:
        raise ValueError(
            f"{prefix} trained {ga.band_suffix(bands)} under {len(per_seed)} seeds "
            f"({', '.join(path.name for path in per_seed)}), and this script resamples "
            "one set of predictions. Whether such a run has one interval per seed or one "
            "over the pooled predictions is not decided; bootstrap.py cannot choose it."
        )


def load_final_result(experiment: int, bands=None) -> FinalResult:
    """Everything about the published model whose result is being quoted.

    Raises:
        ValueError: If the triplet cannot be resolved, or its files cannot be read.
    """
    selected = final_bands(experiment, bands)
    paths = bootstrap_paths(experiment, selected)
    refuse_a_multi_seed_result(experiment, selected, paths["predictions"])
    try:
        nanometres = json.loads(paths["metrics"].read_text())["selected_wavelengths_nm"]
    except (OSError, json.JSONDecodeError, KeyError) as unreadable:
        raise ValueError(
            f"cannot read the wavelengths from {paths['metrics'].name}: {unreadable}"
        ) from unreadable
    return FinalResult(
        experiment,
        selected,
        list(nanometres),
        paths["predictions"],
        read_predictions(paths["predictions"]),
    )


# --- The output -----------------------------------------------------------


# What a rerun has to match for it to be a replay; the rule itself is ga.py's.
# `weighted_f1` is in here as well as the settings, because it is how a
# predictions file that has changed underneath a recorded interval shows up:
# same bands, same seed, same resamples, different number.
RECORDED_SETTINGS = ("selected_band_indices", "resamples", "seed", "weighted_f1")


def run_settings(result: FinalResult, *, resamples: int, seed: int) -> dict:
    """What a rerun has to match to be a replay rather than a different interval."""
    return {
        "selected_band_indices": [int(band) for band in result.bands],
        "resamples": resamples,
        "seed": seed,
        "weighted_f1": weighted_f1(result.predictions),
    }


def write_bootstrap(
    path: Path, result: FinalResult, settings: dict, interval: Interval, table: list[dict]
) -> None:
    """The interval, what produced it, and the capture it came apart on.

    Like the final metrics, it carries no timestamp, so two runs of the same
    experiment, triplet, seed and resample count in the same pinned environment
    write the same bytes.  `libraries` is what says which environment that was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": result.experiment,
                "primary_metric": "weighted_f1",
                "predictions": result.source.name,
                "selected_wavelengths_nm": result.nanometres,
                "resample_unit": "acquisition",
                "interval_method": "percentile",
                "confidence_level": config.BOOTSTRAP_CONFIDENCE_LEVEL,
                "weighted_f1_ci95": [interval.low, interval.high],
                "macro_f1": macro_f1(result.predictions),
                "accuracy": accuracy(result.predictions),
                "test_cropped_hypercube_count": len(result.predictions),
                "test_acquisition_count": len(result.predictions.captures),
                "per_acquisition": table,
                "libraries": {name: version(name) for name in LIBRARIES},
                **settings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


# --- One interval ---------------------------------------------------------


def bootstrap_command(
    experiment: int,
    *,
    bands=None,
    resamples: int = config.BOOTSTRAP_RESAMPLES,
    seed: int = config.BOOTSTRAP_SEED,
    verbose: bool = True,
) -> None:
    """Resample the test predictions of one experiment and write the interval.

    Raises:
        ValueError: If the experiment has no final model, if the triplet is
            ambiguous, if its predictions are missing, or if an interval is
            already recorded under other settings.
    """
    result = load_final_result(experiment, bands)
    paths = bootstrap_paths(experiment, result.bands)
    settings = run_settings(result, resamples=resamples, seed=seed)
    # Before the 5000 draws: a refusal should not cost the work it throws away.
    ga.refuse_a_rerun_that_would_not_reproduce(paths["bootstrap"], settings, RECORDED_SETTINGS)

    interval = bootstrap_interval(result.predictions, resamples=resamples, seed=seed)
    table = per_acquisition_scores(result.predictions)
    write_bootstrap(paths["bootstrap"], result, settings, interval, table)

    if verbose:
        captures = len(result.predictions.captures)
        print(f"candidate      {result.bands}")
        print("wavelengths_nm " + ", ".join(f"{value}" for value in result.nanometres))
        print(f"test crops     {len(result.predictions)} in {captures} acquisitions")
        print(
            f"weighted F1 {interval.point:.3f}  "
            f"95% CI [{interval.low:.3f}, {interval.high:.3f}]  "
            f"({resamples} resamples of {captures} acquisitions, seed {seed})"
        )
        for row in table:
            print(
                f"  {row['majority_class']}  {row['cropped_hypercube_count']:3d} crops  "
                f"weighted F1 {row['weighted_f1']:.3f}  accuracy {row['accuracy']:.2f}  "
                f"{row['acquisition_id'][:12]}"
            )
        print(f"bootstrap      {paths['bootstrap']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", type=int, help="the experiment whose test result to quote")
    parser.add_argument(
        "--bands",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        help="the triplet, when the experiment has more than one final model",
    )
    parser.add_argument(
        "--resamples", type=int, default=config.BOOTSTRAP_RESAMPLES, help="bootstrap draws"
    )
    parser.add_argument("--seed", type=int, default=config.BOOTSTRAP_SEED, help="the rng seed")
    arguments = parser.parse_args(argv)
    try:
        bootstrap_command(
            arguments.experiment,
            bands=arguments.bands,
            resamples=arguments.resamples,
            seed=arguments.seed,
        )
    except ValueError as refusal:
        print(f"bootstrap.py: {refusal}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
