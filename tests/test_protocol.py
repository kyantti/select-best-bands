"""What makes the result defensible, and nothing else.

These tests watch the protocol's observable behaviour: which crops each step
sees, which values come out, and what is refused.  They do not watch how any
of it is written.
"""

import csv

import pytest

import config
from cnn.data_setup import PARTITION_FIELDS, load_cropped_hypercubes, load_partitions
from split_dataset import assign_partitions

REFERENCE_MANIFEST = config.ROOT / "tests" / "data" / "reference_cropped_hypercubes.csv"
REFERENCE_PARTITIONS = config.ROOT / "tests" / "data" / "reference_evaluation_partitions.csv"


def synthetic_manifest(class_counts=(6, 6, 6, 6), crops_per_acquisition=3):
    """A CroppedHypercube manifest of `sum(class_counts)` acquisitions."""
    rows = []
    for severity, count in zip(config.CLASS_NAMES, class_counts, strict=True):
        for acquisition in range(count):
            acquisition_id = f"acq-{severity}-{acquisition:02d}"
            for crop in range(crops_per_acquisition):
                rows.append(
                    {
                        "schema_version": "1",
                        "cropped_hypercube_id": f"{acquisition_id}-crop-{crop}",
                        "crop_id": f"{acquisition_id}-source-{crop}",
                        "acquisition_id": acquisition_id,
                        "severity_class": severity,
                        "spectral_axis_id": "axis",
                    }
                )
    return rows


def split(source_rows, **overrides):
    """`assign_partitions` with the real experiment settings unless overridden."""
    settings = {
        "seed": config.PARTITION_SEED,
        "test_proportion": config.TEST_PROPORTION,
        "validation_proportion": config.VALIDATION_PROPORTION,
    } | overrides
    return assign_partitions(source_rows, **settings)


def assignments_by_acquisition(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["acquisition_id"], set()).add(
            (row["partition"], row["selection_partition"])
        )
    return grouped


def test_split_keeps_every_acquisition_whole_and_every_crop_assigned():
    source = synthetic_manifest()
    rows = split(source)

    assert len(rows) == len(source)
    assert [row["cropped_hypercube_id"] for row in rows] == [
        row["cropped_hypercube_id"] for row in source
    ]
    for acquisition_id, assignments in assignments_by_acquisition(rows).items():
        assert len(assignments) == 1, f"{acquisition_id} crosses a boundary"


def test_split_puts_every_class_on_both_sides_of_both_boundaries():
    rows = split(synthetic_manifest())

    for group in (("train", "train"), ("train", "validation"), ("test", "")):
        present = {
            row["severity_class"]
            for row in rows
            if (row["partition"], row["selection_partition"]) == group
        }
        assert present == set(config.CLASS_NAMES), f"{group} is missing a class"


def test_split_reproduces_the_reference_partition():
    with REFERENCE_PARTITIONS.open(newline="") as source:
        expected = list(csv.DictReader(source))

    rows = split(load_cropped_hypercubes(REFERENCE_MANIFEST))

    assert rows == expected


@pytest.mark.parametrize(
    ("group", "acquisitions", "crops"),
    [(("train", "train"), 22, 708), (("train", "validation"), 6, 160), (("test", ""), 8, 256)],
    ids=["train-fit", "validation", "test"],
)
def test_split_of_the_reference_manifest_has_the_recorded_sizes(group, acquisitions, crops):
    rows = [
        row
        for row in split(load_cropped_hypercubes(REFERENCE_MANIFEST))
        if (row["partition"], row["selection_partition"]) == group
    ]

    assert len(rows) == crops
    assert len({row["acquisition_id"] for row in rows}) == acquisitions


@pytest.mark.skipif(
    not config.HYPERCUBES_MANIFEST.exists(), reason="data/ is gitignored and absent here"
)
def test_split_reads_the_same_manifest_the_reference_partition_came_from():
    """The reproduction only means something while the input is the recorded one."""
    assert config.HYPERCUBES_MANIFEST.read_bytes() == REFERENCE_MANIFEST.read_bytes()


def test_split_fails_when_a_class_has_too_few_acquisitions_to_stratify():
    source = synthetic_manifest(class_counts=(6, 6, 6, 1))

    with pytest.raises(ValueError, match="Acquisition-grouped train/test split") as failure:
        split(source)
    # This one is refused outright, not caught afterwards by the count guard.
    assert "at least one Acquisition per SeverityClass" not in str(failure.value)


def test_split_fails_rather_than_silently_dropping_a_class_from_one_side():
    """scikit-learn can return a split with a rare class on one side only."""
    source = synthetic_manifest(class_counts=(20, 20, 20, 2))

    with pytest.raises(
        ValueError, match="at least one Acquisition per SeverityClass on each side"
    ):
        split(source, test_proportion=0.05)


def reference_partition_rows():
    with REFERENCE_PARTITIONS.open(newline="") as source:
        return list(csv.DictReader(source))


def leak_acquisition_across_train_test(rows):
    """Move one crop of a test Acquisition into train, leaving its siblings behind."""
    victim = next(row for row in rows if row["partition"] == "test")
    victim["partition"] = "train"
    victim["selection_partition"] = "train"


def leak_crop_across_validation(rows):
    """One crop cut in two, with the halves on either side of the inner boundary.

    The twin belongs to an Acquisition that is itself consistently in train-fit,
    so only the crop-level check can refuse this.
    """
    victim = next(row for row in rows if row["selection_partition"] == "validation")
    host = next(row for row in rows if row["selection_partition"] == "train")
    twin = dict(host)
    twin["cropped_hypercube_id"] = f"{victim['cropped_hypercube_id']}-twin"
    twin["crop_id"] = victim["crop_id"]
    rows.append(twin)


def give_a_test_row_a_selection_partition(rows):
    next(row for row in rows if row["partition"] == "test")["selection_partition"] = "validation"


def drop_the_validation_group(rows):
    for row in rows:
        if row["selection_partition"] == "validation":
            row["selection_partition"] = "train"


def drop_the_test_group(rows):
    for row in rows:
        if row["partition"] == "test":
            row["partition"] = "train"
            row["selection_partition"] = "train"


def invent_a_partition_name(rows):
    rows[0]["partition"] = "holdout"


def duplicate_a_hypercube(rows):
    rows.append(dict(rows[0]))


def duplicate_a_crop(rows):
    twin = dict(rows[0])
    twin["cropped_hypercube_id"] = f"{twin['cropped_hypercube_id']}-twin"
    rows.append(twin)


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (leak_acquisition_across_train_test, "Acquisition '.*' crosses evaluation boundaries"),
        (leak_crop_across_validation, "crop '.*' crosses evaluation boundaries"),
        (give_a_test_row_a_selection_partition, "cannot have a selection partition"),
        (drop_the_validation_group, "no validation records"),
        (drop_the_test_group, "no held-out test records"),
        (invent_a_partition_name, "invalid evaluation partition"),
        (duplicate_a_hypercube, "duplicate CroppedHypercube identity"),
        (duplicate_a_crop, "duplicate crop identity"),
    ],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_split_manifest_that_leaks_or_is_incomplete_is_rejected(tmp_path, corrupt, message):
    rows = reference_partition_rows()
    corrupt(rows)
    path = tmp_path / "evaluation_partitions.csv"
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=PARTITION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=message):
        load_partitions(path)


def test_split_manifest_of_the_reference_partition_loads():
    rows = load_partitions(REFERENCE_PARTITIONS)

    assert len(rows) == 1124
    acquisitions = assignments_by_acquisition(rows)
    assert len(acquisitions) == 36
