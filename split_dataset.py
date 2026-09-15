"""Write the evaluation partitions: which Acquisitions train, validate and test.

Two Acquisition-grouped, SeverityClass-stratified splits.  The outer one holds
out `TEST_PROPORTION` of the Acquisitions as test, read once by the final model.
The inner one sets aside `VALIDATION_PROPORTION` of the train Acquisitions as
the validation partition that scores band selection, so no test capture informs
the bands it is later scored with.  Every crop follows its Acquisition.

Usage: uv run python split_dataset.py
"""

import csv
from collections import Counter

from sklearn.model_selection import train_test_split

import config
from cnn.data_setup import PARTITION_FIELDS, load_cropped_hypercubes, validate_partition_rows


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
        raise ValueError(
            f"cannot create SeverityClass-stratified Acquisition-grouped {boundary} split: {exc}"
        ) from exc
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
        raise ValueError(
            f"cannot create SeverityClass-stratified Acquisition-grouped {boundary} split: "
            f"requires at least one Acquisition per SeverityClass on each side; found {counts}"
        )
    return sorted(kept), sorted(held)


def assign_partitions(
    source_rows: list[dict[str, str]],
    *,
    seed: int,
    test_proportion: float,
    validation_proportion: float,
) -> list[dict[str, str]]:
    """Assign every crop of the manifest to train-fit, validation or test.

    The manifest's row order is preserved, so the output can be compared with a
    recorded partition line by line.
    """
    acquisition_classes = {}
    for row in source_rows:
        acquisition_classes[row["acquisition_id"]] = row["severity_class"]
    acquisition_ids = sorted(acquisition_classes)
    train_ids, test_ids = stratified_split(
        acquisition_ids,
        acquisition_classes,
        proportion=test_proportion,
        seed=seed,
        boundary="train/test",
    )
    fit_ids, validation_ids = stratified_split(
        train_ids,
        acquisition_classes,
        proportion=validation_proportion,
        seed=seed,
        boundary="train/validation",
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


def summarize(rows: list[dict[str, str]]) -> None:
    """Print one line per group with its Acquisitions, crops and class counts."""
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


def main() -> None:
    manifest = config.HYPERCUBES_MANIFEST
    output = config.PARTITIONS
    source_rows = load_cropped_hypercubes(manifest)
    rows = assign_partitions(
        source_rows,
        seed=config.PARTITION_SEED,
        test_proportion=config.TEST_PROPORTION,
        validation_proportion=config.VALIDATION_PROPORTION,
    )
    write_partitions(output, rows)
    print(f"{manifest}: {len(source_rows)} crops")
    summarize(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
