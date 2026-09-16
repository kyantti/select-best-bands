"""Ask whether the weak test acquisitions are a data problem, before tuning anything.

    uv run python analyze_test_errors.py N
    uv run python analyze_test_errors.py N --bands R G B

Two of the eight test captures of experiment 21 sit at 0.689 weighted F1 against
0.857-0.951 for the other six, and the errors of the whole test set include
seventeen C0<->C3 confusions, which are the extreme classes and should be the
easy pair to tell apart.  Either those two captures are broken -- overexposed,
badly segmented, mislabelled -- or they are simply hard.  Every later
improvement is worth nothing if it is tuned against the first case, so this runs
first, and it runs on what is already on disk: the per-crop predictions
`train_final.py` wrote, the partition manifest, and the crops themselves.

No model runs here and no GPU is touched.  The held-out crops are opened, which
costs about a minute, but they are only measured and drawn: nothing is scored
that `train_final.py` did not already score.

The conclusion does not live in the output.  It goes, in prose, in the ticket
that asked for this.
"""

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

import bootstrap
import config
import ga
from cnn.data_setup import load_partition_crops, load_spectral_axis, prepare_model_input, wavelengths_of

matplotlib.use("Agg")  # a figure on a headless machine, like every other step
import matplotlib.pyplot as plt  # noqa: E402

Candidate = tuple[int, int, int]

# How many of the worst captures the comparison and the figure are about.  Two
# is what the test set of experiment 21 has: 0.689 twice, then 0.857 upwards.
WEAK_ACQUISITION_COUNT = 2
WEAK, REFERENCE = "weak", "reference"

# Where exposure is measured, in nm, half-open.  Named ranges rather than the
# three selected bands because a lamp or an integration time moves the whole
# spectrum, and a range that the SpectralAxis does not cover is simply dropped.
BAND_RANGES = (
    ("visible", 380.0, 700.0),
    ("red_edge", 700.0, 780.0),
    ("nir", 780.0, 1400.0),
)

ERROR_GRID_COLUMNS = 6
BACKGROUND_GREY = 0.72  # outside the fig, as in `check_data.py`


# --- Where the errors fall -------------------------------------------------


def confusion_counts(predictions: bootstrap.TestPredictions) -> Counter:
    """How many crops of each actual class were called each predicted class.

    Keyed by `(actual, predicted)` over the four class names, the diagonal
    included, so one acquisition's row of the table is complete and a pair that
    never happened reads as zero rather than as a missing key.
    """
    seen = Counter(zip(predictions.actual.tolist(), predictions.predicted.tolist(), strict=True))
    return Counter(
        {
            (actual, predicted): seen[(actual, predicted)]
            for actual in config.CLASS_NAMES
            for predicted in config.CLASS_NAMES
        }
    )


def error_count(counts: Counter) -> int:
    """The crops of a confusion table that landed off its diagonal."""
    return sum(count for (actual, predicted), count in counts.items() if actual != predicted)


def within_acquisition(
    predictions: bootstrap.TestPredictions, capture: str
) -> bootstrap.TestPredictions:
    """The predictions of one capture."""
    return predictions.within(np.flatnonzero(predictions.acquisition == capture))


def weak_acquisitions(predictions: bootstrap.TestPredictions) -> list[str]:
    """The `WEAK_ACQUISITION_COUNT` worst captures by weighted F1, worst first.

    The order is the bootstrap's own per-capture table, ties broken by identity,
    so which captures are called weak does not depend on dictionary order.
    """
    table = bootstrap.per_acquisition_scores(predictions)
    return [row["acquisition_id"] for row in table[:WEAK_ACQUISITION_COUNT]]


# --- What the crops of a capture look like ---------------------------------


@dataclass(frozen=True)
class BandRange:
    """A named stretch of the SpectralAxis, and the bands of it this axis has."""

    name: str
    indices: list[int]


def ranges_in_axis(wavelengths: list[float]) -> list[BandRange]:
    """The band indices of each `BAND_RANGES` entry the SpectralAxis covers.

    A range with no band on this axis is left out instead of reported as empty.

    Raises:
        ValueError: If the axis covers none of them at all.
    """
    covered = []
    for name, low, high in BAND_RANGES:
        indices = [
            index for index, value in enumerate(wavelengths) if low <= value < high
        ]
        if indices:
            covered.append(BandRange(name, indices))
    if not covered:
        raise ValueError(
            f"the SpectralAxis covers none of the band ranges {[name for name, _, _ in BAND_RANGES]}"
        )
    return covered


def measurement_fields(ranges: list[BandRange]) -> tuple[str, ...]:
    """The columns `acquisition_measurements` fills, in the order it fills them."""
    return (
        *(
            f"{band_range.name}_reflectance_{statistic}"
            for band_range in ranges
            for statistic in ("mean", "std")
        ),
        "foreground_fraction",
        "foreground_px_mean",
        "crop_area_px_mean",
    )


def comparison_fields(ranges: list[BandRange]) -> tuple[str, ...]:
    """What the weak captures are compared against the others on.

    The crop count leads, because it is not a property of the crops: a capture
    with more figs in front of the camera contributes more crops to the test
    set, and whether the weak ones are also the big ones is the first thing to
    rule out.
    """
    return ("cropped_hypercube_count", *measurement_fields(ranges))


def acquisition_measurements(crops, ranges: list[BandRange]) -> dict[str, dict]:
    """What can be measured about a capture from its crops alone, per capture.

    Exposure and illumination first: the mean and the spread, over the fig
    pixels of the whole capture, of the reflectance averaged across each band
    range.  Background pixels are excluded, because masked-out area would pull
    every capture towards zero by the amount of table it happens to show.

    Then segmentation: what fraction of the crop the mask calls fig, how many
    pixels that is, and how big the crops are.  A capture whose masks caught the
    tray, or whose crops are half the size of everyone else's, shows up here and
    not in any metric.
    """
    by_acquisition: dict[str, list] = {}
    for hypercube in crops:
        by_acquisition.setdefault(hypercube.acquisition_id, []).append(hypercube)

    measured = {}
    for capture, of_capture in by_acquisition.items():
        row = {}
        for band_range in ranges:
            pixels = np.concatenate(
                [
                    hypercube.reflectance[:, :, band_range.indices][
                        hypercube.foreground_mask
                    ].mean(axis=1)
                    for hypercube in of_capture
                ]
            )
            row[f"{band_range.name}_reflectance_mean"] = float(pixels.mean())
            row[f"{band_range.name}_reflectance_std"] = float(pixels.std())
        areas = np.array(
            [hypercube.foreground_mask.size for hypercube in of_capture], dtype=np.float64
        )
        foreground = np.array(
            [int(hypercube.foreground_mask.sum()) for hypercube in of_capture], dtype=np.float64
        )
        row["foreground_fraction"] = float(foreground.sum() / areas.sum())
        row["foreground_px_mean"] = float(foreground.mean())
        row["crop_area_px_mean"] = float(areas.mean())
        measured[capture] = row
    return measured


# --- The table -------------------------------------------------------------


def confusion_fields() -> tuple[str, ...]:
    """`C0_as_C3`: the sixteen columns of one capture's confusion table."""
    return tuple(
        f"{actual}_as_{predicted}"
        for actual in config.CLASS_NAMES
        for predicted in config.CLASS_NAMES
    )


def analysis_fields(ranges: list[BandRange]) -> tuple[str, ...]:
    """Every column of the analysis table, in the order it is written."""
    return (
        "acquisition_id",
        "group",
        "selected_bands",
        "majority_class",
        "cropped_hypercube_count",
        "weighted_f1",
        "accuracy",
        "error_count",
        *confusion_fields(),
        *measurement_fields(ranges),
    )


def analysis_rows(
    predictions: bootstrap.TestPredictions,
    measurements: dict[str, dict],
    bands: Candidate,
) -> list[dict]:
    """One row per test capture, worst first: its score, its errors, its crops.

    The score columns are the bootstrap's own, not recomputed here, so this
    table and the interval can never disagree about which capture is weak.

    Raises:
        ValueError: If a capture of the predictions has no measured crops.
    """
    scored = bootstrap.per_acquisition_scores(predictions)
    weak = set(weak_acquisitions(predictions))
    rows = []
    for row in scored:
        capture = row["acquisition_id"]
        if capture not in measurements:
            raise ValueError(
                f"no crop of acquisition {capture[:12]} was loaded, so it cannot be measured"
            )
        counts = confusion_counts(within_acquisition(predictions, capture))
        rows.append(
            {
                "acquisition_id": capture,
                "group": WEAK if capture in weak else REFERENCE,
                "selected_bands": ga.band_suffix(bands),
                "majority_class": row["majority_class"],
                "cropped_hypercube_count": row["cropped_hypercube_count"],
                "weighted_f1": row["weighted_f1"],
                "accuracy": row["accuracy"],
                "error_count": error_count(counts),
                **{
                    f"{actual}_as_{predicted}": counts[(actual, predicted)]
                    for actual in config.CLASS_NAMES
                    for predicted in config.CLASS_NAMES
                },
                **measurements[capture],
            }
        )
    return rows


def write_analysis(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    """Write the table, at full precision and with no timestamp, so a rerun replays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def refuse_an_analysis_of_another_triplet(path: Path, bands: Candidate) -> None:
    """Refuse to replace one triplet's analysis with another's under the same name.

    The two outputs are named after the experiment alone, as the ticket asks,
    so an experiment with two final models would otherwise have its table
    silently rewritten by whichever triplet ran last.

    Raises:
        ValueError: If a table is there and was written for other bands, or
            cannot be read to find out.
    """
    if not path.exists():
        return
    try:
        rows = list(csv.DictReader(path.open(newline="")))
    except OSError as unreadable:
        raise ValueError(
            f"cannot read {path.name} to check which triplet it is about: {unreadable}"
        ) from unreadable
    recorded = {row.get("selected_bands") for row in rows}
    suffix = ga.band_suffix(bands)
    if recorded and recorded != {suffix}:
        raise ValueError(
            f"{path.name} was recorded for bands {', '.join(sorted(name or '?' for name in recorded))}; "
            f"rewriting it for {suffix} would replace one result under out/ with another."
        )


# --- The figure ------------------------------------------------------------


@dataclass(frozen=True)
class CropPrediction:
    """One held-out crop, what it is, and what the final model called it.

    The interval needs three columns and drops the identity; a figure has to
    put a panel next to an artifact, so this carries it.
    """

    cropped_hypercube_id: str
    acquisition_id: str
    actual: str
    predicted: str

    @property
    def misread(self) -> bool:
        return self.actual != self.predicted


def read_records(path: Path) -> list[CropPrediction]:
    """Every test crop of a predictions file, with the identity the interval drops.

    `bootstrap.read_predictions` keeps the three columns a score needs; the
    figure also needs to know which artifact each row is about, so the file is
    read again here rather than teaching the interval about crops.

    Raises:
        ValueError: If the file is not there, or is not a predictions file.
    """
    if not path.exists():
        raise ValueError(
            f"{path.name} is not under {path.parent}; run train_final.py for this triplet first."
        )
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    columns = ("cropped_hypercube_id", "acquisition_id", "actual", "predicted")
    if not rows or any(column not in rows[0] for column in columns):
        raise ValueError(f"{path.name} does not have the columns {columns} of a predictions file")
    return [CropPrediction(*(row[column] for column in columns)) for row in rows]


def misclassified(records: list[CropPrediction], acquisitions) -> list[CropPrediction]:
    """Every misread crop of the named captures, in the order of the predictions."""
    wanted = list(acquisitions)
    return [
        record
        for record in records
        if record.acquisition_id in wanted and record.misread
    ]


def draw_errors(
    entries: list[CropPrediction], crops_by_id: dict, bands, wavelengths: list[float], path: Path
) -> Path:
    """One panel per misread crop, titled with what it is and what it was called.

    The panels hold the raw reflectance of the three selected bands, resized the
    way the network resizes it but not normalized: normalization is one affine
    map shared by the whole dataset, so a capture that is dark stays dark in the
    model input, and the point of the grid is to see whether these captures are
    dark, blown out, or badly cut.  One scale over the fig pixels of the whole
    grid keeps the captures comparable; the masked background is painted flat
    grey outside that scale, so an empty channel cannot pass for no fig.

    Raises:
        ValueError: If there is nothing to draw, or a crop of the grid was not
            loaded.
    """
    if not entries:
        raise ValueError("there are no misread crops to draw")
    missing = [entry.cropped_hypercube_id for entry in entries if entry.cropped_hypercube_id not in crops_by_id]
    if missing:
        raise ValueError(f"no crop was loaded for {missing[0]}")
    identity = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # raw reflectance, resized as the model resizes
    prepared = [
        prepare_model_input(
            crops_by_id[entry.cropped_hypercube_id], list(bands), *identity, size=config.IMAGE_SIZE
        )
        for entry in entries
    ]
    images = [image.permute(1, 2, 0).numpy() for image, _ in prepared]
    masks = [mask.numpy() for _, mask in prepared]
    foreground = np.concatenate([image[mask] for image, mask in zip(images, masks, strict=True)])
    low, high = float(foreground.min()), float(foreground.max())
    span = high - low if high > low else 1.0
    scaled = []
    for image, mask in zip(images, masks, strict=True):
        panel = np.clip((image - low) / span, 0, 1)
        panel[~mask] = BACKGROUND_GREY
        scaled.append(panel)

    columns = min(ERROR_GRID_COLUMNS, len(entries))
    rows = (len(entries) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 1.9 * rows), squeeze=False)
    panels = axes.flatten()
    for axis, entry, image in zip(panels, entries, scaled, strict=False):
        axis.imshow(image)
        axis.set_title(
            f"{entry.actual} read as {entry.predicted}\n"
            f"{entry.acquisition_id[:8]} / {entry.cropped_hypercube_id[:8]}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in panels[len(entries) :]:
        axis.axis("off")

    nanometres = " / ".join(f"{value:g} nm" for value in wavelengths_of(tuple(bands), wavelengths))
    captures = sorted({entry.acquisition_id[:8] for entry in entries})
    figure.suptitle(
        f"CropPrediction test crops of the weak acquisitions {', '.join(captures)} — bands "
        f"{'/'.join(str(band) for band in bands)} = {nanometres} as R/G/B\n"
        "raw reflectance, one scale over the fig pixels of the whole grid"
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, hspace=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# --- The report ------------------------------------------------------------


def analysis_paths(experiment: int) -> dict[str, Path]:
    """The table and the figure this analysis owns.

    Named after the experiment and not after the triplet, as the ticket asks;
    `refuse_an_analysis_of_another_triplet` is what keeps that unambiguous.
    """
    prefix = ga.experiment_prefix(experiment)
    return {
        "analysis": config.TABLES_DIR / f"{prefix}_test_error_analysis.csv",
        "figure": config.FIGURES_DIR / f"{prefix}_test_errors.png",
    }


def confusion_summary(row: dict) -> str:
    """The off-diagonal cells of one row, worst first: `C0->C3 11, C0->C2 6`."""
    errors = [
        (row[f"{actual}_as_{predicted}"], actual, predicted)
        for actual in config.CLASS_NAMES
        for predicted in config.CLASS_NAMES
        if actual != predicted and row[f"{actual}_as_{predicted}"]
    ]
    errors.sort(key=lambda cell: (-cell[0], cell[1], cell[2]))
    return ", ".join(f"{actual}->{predicted} {count}" for count, actual, predicted in errors) or "-"


def extreme_confusions(predictions: bootstrap.TestPredictions, weak) -> dict:
    """How many crops crossed the whole severity range, and where they fell.

    The first and last class are the extremes: a C0 called C3 is the error the
    ticket is about, because those two should be the easy pair to tell apart.
    Whether they sit in the weak captures or run through all eight is the
    difference between a broken capture and a model that is not ordinal.
    """
    mildest, severest = config.CLASS_NAMES[0], config.CLASS_NAMES[-1]
    counts = confusion_counts(predictions)
    in_weak = confusion_counts(
        predictions.within(np.isin(predictions.acquisition, list(weak)))
    )
    pairs = ((mildest, severest), (severest, mildest))
    return {
        "pair": f"{mildest}<->{severest}",
        "total": sum(counts[pair] for pair in pairs),
        "in_weak": sum(in_weak[pair] for pair in pairs),
        "by_direction": {f"{actual}->{predicted}": counts[(actual, predicted)] for actual, predicted in pairs},
    }


def group_comparison(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    """Each measurement averaged over the weak captures and over the others.

    Two numbers and their difference, which is all a group of two against a
    group of six supports: with eight captures there is no test to run here,
    only a question of whether the weak ones are off the scale of the rest.
    """
    comparison = []
    for field in fields:
        groups = {
            name: [row[field] for row in rows if row["group"] == name and field in row]
            for name in (WEAK, REFERENCE)
        }
        if not all(groups.values()):
            continue
        weak = float(np.mean(groups[WEAK]))
        reference = float(np.mean(groups[REFERENCE]))
        comparison.append(
            {
                "measurement": field,
                WEAK: weak,
                REFERENCE: reference,
                "difference": weak - reference,
            }
        )
    return comparison


def print_confusion(counts: Counter) -> None:
    """The whole test set as a 4x4 block, so the extreme corners are visible."""
    print(f"  {'actual':8}" + "".join(f"{f'as {name}':>8}" for name in config.CLASS_NAMES))
    for actual in config.CLASS_NAMES:
        print(
            f"  {actual:8}"
            + "".join(f"{counts[(actual, predicted)]:8d}" for predicted in config.CLASS_NAMES)
        )


def print_report(result: bootstrap.FinalResult, rows: list[dict], comparison: list[dict]) -> None:
    """Everything the ticket asks to be printed, and nothing that needs a file."""
    predictions = result.predictions
    print(f"candidate      {result.bands}")
    print("wavelengths_nm " + ", ".join(f"{value}" for value in result.nanometres))
    print(
        f"test crops     {len(predictions)} in {len(predictions.captures)} acquisitions, "
        f"{error_count(confusion_counts(predictions))} of them misread"
    )
    print()
    print("per acquisition, worst first")
    print(
        f"  {'group':10} {'class':5} {'crops':>5} {'wF1':>6} {'acc':>5} {'err':>4}  "
        f"{'confusions':38} id"
    )
    for row in rows:
        print(
            f"  {row['group']:10} {row['majority_class']:5} "
            f"{row['cropped_hypercube_count']:5d} {row['weighted_f1']:6.3f} "
            f"{row['accuracy']:5.2f} {row['error_count']:4d}  "
            f"{confusion_summary(row):38} {row['acquisition_id'][:12]}"
        )
    print()
    print("the whole test set")
    print_confusion(confusion_counts(predictions))
    extreme = extreme_confusions(predictions, weak_acquisitions(predictions))
    directions = ", ".join(f"{name} {count}" for name, count in extreme["by_direction"].items())
    print(
        f"  {extreme['pair']} {extreme['total']} of "
        f"{error_count(confusion_counts(predictions))} errors ({directions}); "
        f"{extreme['in_weak']} of them in the {WEAK} acquisitions"
    )
    print()
    weak = [row["acquisition_id"][:8] for row in rows if row["group"] == WEAK]
    others = sum(1 for row in rows if row["group"] == REFERENCE)
    print(f"the crops themselves: weak ({', '.join(weak)}) against the other {others}")
    print(f"  {'measurement':28} {'weak':>10} {'rest':>10} {'difference':>12}")
    for entry in comparison:
        print(
            f"  {entry['measurement']:28} {entry[WEAK]:10.4f} "
            f"{entry[REFERENCE]:10.4f} {entry['difference']:+12.4f}"
        )


# --- One analysis ----------------------------------------------------------


def analyze_command(experiment: int, *, bands=None, verbose: bool = True) -> None:
    """Cross the test predictions of one experiment with its crops, and report.

    Raises:
        ValueError: If the experiment has no final model, if the triplet is
            ambiguous, if its predictions are missing, or if an analysis of
            another triplet is already recorded under the same name.
    """
    result = bootstrap.load_final_result(experiment, bands)
    paths = analysis_paths(experiment)
    # Before the crops are opened: a refusal should not cost the minute it wastes.
    refuse_an_analysis_of_another_triplet(paths["analysis"], result.bands)

    wavelengths = load_spectral_axis(config.SPECTRAL_AXES)
    ranges = ranges_in_axis(wavelengths)
    crops = load_partition_crops(config.HYPERCUBES_MANIFEST, "test", wavelengths, verbose=verbose)
    measurements = acquisition_measurements(crops, ranges)

    rows = analysis_rows(result.predictions, measurements, result.bands)
    fields = analysis_fields(ranges)
    write_analysis(paths["analysis"], rows, fields)

    weak = weak_acquisitions(result.predictions)
    misread = misclassified(read_records(result.source), weak)
    figure = (
        draw_errors(
            misread,
            {hypercube.cropped_hypercube_id: hypercube for hypercube in crops},
            result.bands,
            wavelengths,
            paths["figure"],
        )
        if misread
        else None
    )

    if verbose:
        print_report(result, rows, group_comparison(rows, comparison_fields(ranges)))
        print()
        print(f"analysis       {paths['analysis']}")
        print(f"figure         {figure or 'not drawn: the weak acquisitions have no misread crop'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", type=int, help="the experiment whose test errors to look at")
    parser.add_argument(
        "--bands",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        help="the triplet, when the experiment has more than one final model",
    )
    arguments = parser.parse_args(argv)
    try:
        analyze_command(arguments.experiment, bands=arguments.bands)
    except ValueError as refusal:
        print(f"analyze_test_errors.py: {refusal}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
