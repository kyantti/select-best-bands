"""Write the evaluation partitions: which Acquisitions train, validate and test.

Two Acquisition-grouped, SeverityClass-stratified splits.  The outer one holds
out `TEST_PROPORTION` of the Acquisitions as test, read once by the final model.
The inner one sets aside `VALIDATION_PROPORTION` of the train Acquisitions as
the validation partition that scores band selection, so no test capture informs
the bands it is later scored with.  Every crop follows its Acquisition.

The inner boundary has two policies (`config.VALIDATION_BALANCE`): the default
stratifies by Acquisition, and `crop_count_balanced` instead picks the
admissible subset whose crops are spread most evenly over the classes.

Usage:
  uv run python split_dataset.py
  uv run python split_dataset.py --validation-balance crop_count_balanced
"""

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from sklearn.model_selection import train_test_split

import config
from cnn.data_setup import PARTITION_FIELDS, load_manifest, validate_partition_rows


def unsplittable(boundary: str, detail: str) -> ValueError:
    """The one refusal both policies raise: no split of this shape exists."""
    return ValueError(
        f"cannot create SeverityClass-stratified Acquisition-grouped {boundary} split: {detail}"
    )


def stratified_split(
    identities: list[str],
    classes: dict[str, str],
    *,
    proportion: float,
    seed: int,
    boundary: str,
) -> tuple[list[str], list[str]]:
    """Split identities in two, keeping every class on both sides.

    Raises:
        ValueError: If no stratified split is possible, or if the one scikit-learn
            produced leaves a SeverityClass without an Acquisition on one side.
    """
    labels = [classes[identity] for identity in identities]
    try:
        kept, held = train_test_split(
            identities, test_size=proportion, random_state=seed, stratify=labels
        )
    except ValueError as exc:
        raise unsplittable(boundary, str(exc)) from exc
    kept_counts = Counter(classes[identity] for identity in kept)
    held_counts = Counter(classes[identity] for identity in held)
    unsupported = {
        severity_class: (kept_counts[severity_class], held_counts[severity_class])
        for severity_class in sorted(set(labels))
        if not kept_counts[severity_class] or not held_counts[severity_class]
    }
    if unsupported:
        counts = ", ".join(
            f"{severity_class}={kept}/{held}"
            for severity_class, (kept, held) in unsupported.items()
        )
        raise unsplittable(
            boundary,
            f"requires at least one Acquisition per SeverityClass on each side; found {counts}",
        )
    return sorted(kept), sorted(held)


def held_out_size(total: int, proportion: float) -> int:
    """How many identities a proportion holds out, as `train_test_split` counts them.

    Both policies then publish a validation partition of the same size.
    """
    return math.ceil(total * proportion)


def count_crops_per_class(
    identities: list[str] | tuple[str, ...], classes: dict[str, str], crops: dict[str, int]
) -> dict[str, int]:
    counted = Counter()
    for identity in identities:
        counted[classes[identity]] += crops[identity]
    return dict(counted)


def balance_ratio(counted: dict[str, int]) -> float:
    """The max/min crops per class: 1.0 is even, and higher weighs some classes more."""
    return max(counted.values()) / min(counted.values())


def balanced_split(
    identities: list[str],
    classes: dict[str, str],
    crops: dict[str, int],
    *,
    proportion: float,
    boundary: str,
) -> tuple[list[str], list[str]]:
    """Hold out the subset whose crops are spread most evenly over the classes.

    A deterministic choice, not a random one: every admissible subset of the
    requested size is scored by the max/min crops per class and the minimum
    wins, ties broken by the sorted identities.  Admissible keeps what the
    stratified policy keeps -- at least one identity per class on *both* sides.
    The target ratio is not a filter: the best subset found is published
    whatever it scores, and the caller records the number.

    Raises:
        ValueError: If no subset of that size leaves every class on both sides.
    """
    present = sorted(set(classes[identity] for identity in identities))
    size = held_out_size(len(identities), proportion)
    best = None
    if len(present) <= size <= len(identities) - len(present):
        for held in combinations(sorted(identities), size):
            kept = set(identities) - set(held)
            if len(set(classes[identity] for identity in held)) < len(present):
                continue
            if len(set(classes[identity] for identity in kept)) < len(present):
                continue
            score = (balance_ratio(count_crops_per_class(held, classes, crops)), held)
            if best is None or score < best:
                best = score
    if best is None:
        raise unsplittable(
            boundary,
            f"requires at least one Acquisition per SeverityClass on each side; no {size} of "
            f"the {len(identities)} Acquisitions leave one for each of {', '.join(present)}",
        )
    _, held = best
    return sorted(set(identities) - set(held)), sorted(held)


def validation_split(
    train_ids: list[str],
    classes: dict[str, str],
    crops: dict[str, int],
    *,
    balance: str,
    proportion: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Draw the train/validation boundary under the requested policy."""
    if balance == "acquisition_stratified":
        return stratified_split(
            train_ids, classes, proportion=proportion, seed=seed, boundary="train/validation"
        )
    if balance == "crop_count_balanced":
        return balanced_split(
            train_ids, classes, crops, proportion=proportion, boundary="train/validation"
        )
    raise ValueError(
        f"unknown validation balance {balance!r}: expected one of "
        f"{', '.join(config.VALIDATION_BALANCES)}"
    )


@dataclass(frozen=True)
class ValidationBalance:
    """How evenly a published validation partition is spread over the classes.

    Read from the rows that were written, so it describes the partition the
    search will actually be scored on rather than what a policy intended.
    """

    crops_per_class: dict[str, int]
    ratio: float

    @property
    def reaches_target(self) -> bool:
        return self.ratio <= config.VALIDATION_BALANCE_TARGET_RATIO


def measure_balance(rows: list[dict[str, str]]) -> ValidationBalance:
    """Weigh a published partition's validation crops against each other.

    Raises:
        ValueError: If a SeverityClass has no crop in the validation partition.
    """
    counted = Counter(
        row["severity_class"] for row in rows if row["selection_partition"] == "validation"
    )
    missing = [name for name in config.CLASS_NAMES if not counted[name]]
    if missing:
        raise ValueError(
            f"validation partition has no crop of {', '.join(missing)}: "
            "it cannot be weighed against the other classes"
        )
    counts = {name: counted[name] for name in config.CLASS_NAMES}
    return ValidationBalance(crops_per_class=counts, ratio=balance_ratio(counts))


def assign_partitions(
    source_rows: list[dict[str, str]],
    *,
    seed: int,
    test_proportion: float,
    validation_proportion: float,
    balance: str = config.VALIDATION_BALANCE,
) -> list[dict[str, str]]:
    """Assign every crop of the manifest to train-fit, validation or test.

    The manifest's row order is preserved, so the output can be compared with a
    recorded partition line by line.
    """
    acquisition_classes = {}
    acquisition_crops = Counter()
    for row in source_rows:
        acquisition_classes[row["acquisition_id"]] = row["severity_class"]
        acquisition_crops[row["acquisition_id"]] += 1
    acquisition_ids = sorted(acquisition_classes)
    train_ids, test_ids = stratified_split(
        acquisition_ids,
        acquisition_classes,
        proportion=test_proportion,
        seed=seed,
        boundary="train/test",
    )
    fit_ids, validation_ids = validation_split(
        train_ids,
        acquisition_classes,
        acquisition_crops,
        balance=balance,
        proportion=validation_proportion,
        seed=seed,
    )
    test_set = set(test_ids)
    validation_set = set(validation_ids)
    del fit_ids
    rows = []
    for source in source_rows:
        acquisition_id = source["acquisition_id"]
        if acquisition_id in test_set:
            partition, selection = "test", ""
        elif acquisition_id in validation_set:
            partition, selection = "train", "validation"
        else:
            partition, selection = "train", "train"
        rows.append(
            {
                "schema_version": "1",
                "cropped_hypercube_id": source["cropped_hypercube_id"],
                "crop_id": source["crop_id"],
                "acquisition_id": acquisition_id,
                "severity_class": source["severity_class"],
                "partition": partition,
                "selection_partition": selection,
            }
        )
    return validate_partition_rows(rows)


def write_partitions(path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=PARTITION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


GROUPS = (
    ("train", "train", "train-fit"),
    ("train", "validation", "validation"),
    ("test", "", "test"),
)


def summarize(rows: list[dict[str, str]], *, balance: str) -> None:
    """Print one line per group with its Acquisitions, crops and class counts.

    Then the line the report quotes: which policy drew the train/validation
    boundary and how evenly it spread the crops that the weighted-F1 fitness
    weighs.  The target is a criterion, not a promise, so the line says whether
    the published partition reached it.
    """
    for partition, selection, name in GROUPS:
        group = [
            row
            for row in rows
            if (row["partition"], row["selection_partition"]) == (partition, selection)
        ]
        acquisitions = {row["acquisition_id"] for row in group}
        classes = Counter(row["severity_class"] for row in group)
        per_class = " ".join(
            f"{severity_class}={classes[severity_class]}"
            for severity_class in config.CLASS_NAMES
        )
        print(f"{name:11s} {len(acquisitions):3d} acquisitions {len(group):5d} crops  {per_class}")
    achieved = measure_balance(rows)
    reached = "reached" if achieved.reaches_target else "not reached"
    print(
        f"{balance}: max/min crops per class {achieved.ratio:.2f} "
        f"(target {config.VALIDATION_BALANCE_TARGET_RATIO:.2f}, {reached})"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--validation-balance",
        choices=config.VALIDATION_BALANCES,
        default=config.VALIDATION_BALANCE,
        help="how to draw the train/validation boundary (default: %(default)s)",
    )
    arguments = parser.parse_args(argv)
    manifest = config.HYPERCUBES_MANIFEST
    output = config.PARTITIONS
    source_rows = load_manifest(manifest)
    try:
        rows = assign_partitions(
            source_rows,
            seed=config.PARTITION_SEED,
            test_proportion=config.TEST_PROPORTION,
            validation_proportion=config.VALIDATION_PROPORTION,
            balance=arguments.validation_balance,
        )
        print(f"{manifest}: {len(source_rows)} crops")
        summarize(rows, balance=arguments.validation_balance)
    except ValueError as refusal:
        # Nothing is on disk yet: a partition that cannot be weighed is not written.
        print(f"split_dataset.py: {refusal}", file=sys.stderr)
        return 1
    write_partitions(output, rows)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
