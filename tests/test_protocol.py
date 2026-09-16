"""What makes the result defensible, and nothing else.

These tests watch the protocol's observable behaviour: which crops each step
sees, which values come out, and what is refused.  They do not watch how any
of it is written.
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import analyze_test_errors
import bootstrap
import check_data
import config
import ga
import plot_fitness_evolution
import train_final
from cnn.data_setup import (
    CROPPED_HYPERCUBE_FIELDS,
    PARTITION_FIELDS,
    SPECTRAL_AXIS_FIELDS,
    SelectedBandDataset,
    apply_foreground_normalization,
    fit_foreground_normalization,
    load_hypercubes,
    load_manifest,
    load_partition_crops,
    load_partitions,
    load_spectral_axis,
    prepare_model_input,
    spectral_axis_id,
    wavelengths_of,
)
from cnn.model import candidate_seed, classification_metrics, data_identity
from split_dataset import assign_partitions

BANDS = 5

REFERENCE_MANIFEST = config.ROOT / "tests" / "data" / "reference_cropped_hypercubes.csv"
REFERENCE_PARTITIONS = config.ROOT / "tests" / "data" / "reference_evaluation_partitions.csv"
REFERENCE_SPECTRAL_AXES = config.ROOT / "tests" / "data" / "reference_spectral_axes.csv"


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

    rows = split(load_manifest(REFERENCE_MANIFEST))

    assert rows == expected


@pytest.mark.parametrize(
    ("group", "acquisitions", "crops"),
    [(("train", "train"), 22, 708), (("train", "validation"), 6, 160), (("test", ""), 8, 256)],
    ids=["train-fit", "validation", "test"],
)
def test_split_of_the_reference_manifest_has_the_recorded_sizes(group, acquisitions, crops):
    rows = [
        row
        for row in split(load_manifest(REFERENCE_MANIFEST))
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


# --- Hypercube loading, normalization and augmentation ----------------------


def crop(value, *, height=1, width=1, foreground=None):
    """A reflectance cube whose foreground pixels all hold `value` per band."""
    reflectance = np.zeros((height, width, BANDS), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.bool_) if foreground is None else foreground.copy()
    if foreground is None:
        mask[:] = True
    reflectance[mask] = np.asarray(value, dtype=np.float32)
    return reflectance, mask


def write_dataset(directory, crops, *, axis_id="axis"):
    """Write a CroppedHypercube manifest and its NPZ artifacts; return the manifest path."""
    manifest = directory / "cropped_hypercubes.csv"
    artifacts = directory / "cropped_hypercubes"
    artifacts.mkdir(parents=True, exist_ok=True)
    rows = []
    for identity, acquisition, severity, (reflectance, mask) in crops:
        np.savez_compressed(
            artifacts / f"{identity}.npz", reflectance=reflectance, foreground_mask=mask
        )
        rows.append(
            {
                "schema_version": "1",
                "cropped_hypercube_id": identity,
                "crop_id": f"crop-{identity}",
                "acquisition_id": acquisition,
                "annotation_id": f"annotation-{identity}",
                "severity_class": severity,
                "height": str(reflectance.shape[0]),
                "width": str(reflectance.shape[1]),
                "band_count": str(reflectance.shape[2]),
                "artifact_relative_path": f"cropped_hypercubes/{identity}.npz",
                "artifact_checksum": "unchecked",
                "spectral_axis_id": axis_id,
            }
        )
    with manifest.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=CROPPED_HYPERCUBE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def partition_row(identity, acquisition, severity, partition, selection):
    return {
        "schema_version": "1",
        "cropped_hypercube_id": identity,
        "crop_id": f"crop-{identity}",
        "acquisition_id": acquisition,
        "severity_class": severity,
        "partition": partition,
        "selection_partition": selection,
    }


@pytest.fixture
def tiny_dataset(tmp_path):
    """Two train-fit crops, one validation crop and one test crop.

    The manifest order is deliberately not the alphabetical order of the
    identities, so a loader that sorted them would be caught.
    """
    crops = [
        ("fit-z", "acq-train", "C0", crop(0.2)),
        ("fit-a", "acq-train", "C0", crop(0.6)),
        ("val-a", "acq-validation", "C1", crop(0.4)),
        ("test-a", "acq-test", "C3", crop(0.9)),
    ]
    manifest = write_dataset(tmp_path, crops)
    rows = {
        "fit": [
            partition_row("fit-z", "acq-train", "C0", "train", "train"),
            partition_row("fit-a", "acq-train", "C0", "train", "train"),
        ],
        "validation": [partition_row("val-a", "acq-validation", "C1", "train", "validation")],
        "test": [partition_row("test-a", "acq-test", "C3", "test", "")],
    }
    return manifest, rows


def test_load_hypercubes_never_returns_a_test_identity(tiny_dataset):
    manifest, rows = tiny_dataset

    loaded = load_hypercubes(manifest, rows["fit"] + rows["validation"], verbose=False)

    identities = [hypercube.cropped_hypercube_id for hypercube in loaded]
    assert identities == ["fit-z", "fit-a", "val-a"]
    assert "test-a" not in identities


def test_load_hypercubes_follows_manifest_order_not_the_order_asked_for(tiny_dataset):
    manifest, rows = tiny_dataset
    asked = list(reversed(rows["fit"]))

    loaded = load_hypercubes(manifest, asked, verbose=False)

    # Manifest order, which here is neither the order asked for nor sorted order.
    assert [hypercube.cropped_hypercube_id for hypercube in loaded] == ["fit-z", "fit-a"]


def test_load_hypercubes_rejects_lineage_that_disagrees_with_the_manifest(tiny_dataset):
    manifest, rows = tiny_dataset
    tampered = [dict(rows["fit"][0], severity_class="C3")]

    with pytest.raises(ValueError, match="lineage mismatch"):
        load_hypercubes(manifest, tampered, verbose=False)


def test_load_hypercubes_rejects_an_identity_the_manifest_does_not_have(tiny_dataset):
    manifest, rows = tiny_dataset
    absent = [partition_row("ghost", "acq-train", "C0", "train", "train")]

    with pytest.raises(ValueError, match="missing requested identities"):
        load_hypercubes(manifest, absent, verbose=False)


def test_load_hypercubes_carries_the_class_and_the_spectral_axis(tiny_dataset):
    manifest, rows = tiny_dataset

    loaded = load_hypercubes(manifest, rows["validation"], verbose=False)

    assert loaded[0].severity_class == 1
    assert loaded[0].spectral_axis_id == "axis"
    assert loaded[0].reflectance.dtype == np.float32


def test_load_spectral_axis_gives_the_selected_bands_their_wavelengths():
    wavelengths = load_spectral_axis(REFERENCE_SPECTRAL_AXES)

    assert len(wavelengths) == 448
    # Exactly what the 10 Sep run recorded for the winner. Its artifacts hold
    # 891.02 and 697.05; the thesis prose rounds them by hand to 891,0 and 697,1.
    assert wavelengths_of((366, 262, 225), wavelengths) == (891.02, 747.5, 697.05)


@pytest.mark.parametrize(
    "bands",
    [(1, 1, 2), (1, 2), (1, 2, 448), (-1, 2, 3), (1.0, 2, 3)],
    ids=["repeated", "two", "out-of-range", "negative", "not-an-int"],
)
def test_load_spectral_axis_refuses_a_triplet_that_is_not_three_distinct_bands(bands):
    wavelengths = load_spectral_axis(REFERENCE_SPECTRAL_AXES)

    with pytest.raises(ValueError, match="three distinct in-range"):
        wavelengths_of(bands, wavelengths)


def test_load_spectral_axis_refuses_wavelengths_that_are_not_the_axis_they_claim(tmp_path):
    """The axis id is the content address of its wavelengths, so an edited row shows."""
    rows = list(csv.DictReader(REFERENCE_SPECTRAL_AXES.open(newline="")))
    edited = json.loads(rows[0]["wavelengths_nm"])
    edited[0] = edited[0] + 0.01
    rows[0]["wavelengths_nm"] = json.dumps(edited, separators=(",", ":"))
    path = tmp_path / "spectral_axes.csv"
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="invalid SpectralAxis definition"):
        load_spectral_axis(path)


def test_load_spectral_axis_hashes_to_the_id_the_crops_were_cut_against():
    wavelengths = load_spectral_axis(REFERENCE_SPECTRAL_AXES)
    manifest_row = next(csv.DictReader(REFERENCE_MANIFEST.open(newline="")))

    assert spectral_axis_id(wavelengths) == manifest_row["spectral_axis_id"]


@pytest.mark.skipif(
    not config.SPECTRAL_AXES.exists(), reason="data/ is gitignored and absent here"
)
def test_load_spectral_axis_reads_the_axis_the_reference_was_taken_from():
    assert config.SPECTRAL_AXES.read_bytes() == REFERENCE_SPECTRAL_AXES.read_bytes()


def test_normalization_is_fitted_on_foreground_pixels_only(tmp_path):
    foreground = np.array([[True, False]])
    crops = [
        ("fit-a", "acq", "C0", crop(0.2, height=1, width=2, foreground=foreground)),
        ("fit-b", "acq", "C0", crop(0.6, height=1, width=2, foreground=foreground)),
    ]
    manifest = write_dataset(tmp_path, crops)
    rows = [
        partition_row("fit-a", "acq", "C0", "train", "train"),
        partition_row("fit-b", "acq", "C0", "train", "train"),
    ]
    hypercubes = load_hypercubes(manifest, rows, verbose=False)

    mean, std = fit_foreground_normalization(hypercubes, (0, 1, 2))

    # Each crop has one foreground pixel and one zero background pixel; counting
    # the background in would drag the mean to 0.2 and the population std to 0.28.
    assert mean == pytest.approx((0.4, 0.4, 0.4))
    assert std == pytest.approx((0.2, 0.2, 0.2))


def test_normalization_fails_on_a_zero_variance_channel(tmp_path):
    manifest = write_dataset(tmp_path, [("fit-a", "acq", "C0", crop(0.4))])
    rows = [partition_row("fit-a", "acq", "C0", "train", "train")]
    hypercubes = load_hypercubes(manifest, rows, verbose=False)

    with pytest.raises(ValueError, match="zero-variance channel"):
        fit_foreground_normalization(hypercubes, (0, 1, 2))


def test_normalization_keeps_the_background_at_zero(tmp_path):
    foreground = np.array([[True, False]])
    crops = [
        ("fit-a", "acq", "C0", crop(0.2, height=1, width=2, foreground=foreground)),
        ("fit-b", "acq", "C0", crop(0.6, height=1, width=2, foreground=foreground)),
    ]
    manifest = write_dataset(tmp_path, crops)
    rows = [
        partition_row("fit-a", "acq", "C0", "train", "train"),
        partition_row("fit-b", "acq", "C0", "train", "train"),
    ]
    hypercubes = load_hypercubes(manifest, rows, verbose=False)
    mean, std = fit_foreground_normalization(hypercubes, (0, 1, 2))

    image, mask = apply_foreground_normalization(hypercubes[0], (0, 1, 2), mean, std)

    assert image.dtype == np.float32
    assert np.array_equal(mask, hypercubes[0].foreground_mask)
    assert (image[~mask] == 0).all()
    assert image[mask][0] == pytest.approx(-1.0)  # 0.2 is one population std below 0.4


def augmentable(tmp_path):
    """One crop whose foreground is its left column, plus fitted statistics."""
    foreground = np.array([[True, False], [True, False]])
    crops = [
        ("fit-a", "acq", "C0", crop(0.2, height=2, width=2, foreground=foreground)),
        ("fit-b", "acq", "C0", crop(0.6, height=2, width=2, foreground=foreground)),
    ]
    manifest = write_dataset(tmp_path, crops)
    rows = [
        partition_row("fit-a", "acq", "C0", "train", "train"),
        partition_row("fit-b", "acq", "C0", "train", "train"),
    ]
    hypercubes = load_hypercubes(manifest, rows, verbose=False)
    return hypercubes, fit_foreground_normalization(hypercubes, (0, 1, 2))


def test_augmentation_moves_the_mask_with_the_image(tmp_path):
    hypercubes, (mean, std) = augmentable(tmp_path)

    plain, plain_mask = prepare_model_input(
        hypercubes[0], (0, 1, 2), mean, std, size=(2, 2)
    )
    flipped, flipped_mask = prepare_model_input(
        hypercubes[0], (0, 1, 2), mean, std, size=(2, 2), horizontal_flip=True
    )

    assert torch.equal(flipped_mask, torch.flip(plain_mask, dims=[1]))
    assert torch.equal(flipped, torch.flip(plain, dims=[2]))


@pytest.mark.parametrize(
    "augmentation",
    [
        {"horizontal_flip": True},
        {"vertical_flip": True},
        {"rotation_degrees": 12.0},
        {"horizontal_flip": True, "vertical_flip": True, "rotation_degrees": -7.5},
    ],
    ids=["hflip", "vflip", "rotation", "all"],
)
def test_augmentation_leaves_no_reflectance_outside_the_mask(tmp_path, augmentation):
    hypercubes, (mean, std) = augmentable(tmp_path)

    image, mask = prepare_model_input(
        hypercubes[0], (0, 1, 2), mean, std, size=(8, 8), **augmentation
    )

    assert mask.any()
    assert (image[:, ~mask] == 0).all()


def test_augmentation_repeats_when_the_torch_generator_is_seeded_the_same(tmp_path):
    hypercubes, (mean, std) = augmentable(tmp_path)
    dataset = SelectedBandDataset(hypercubes, (0, 1, 2), mean, std, (8, 8), training=True)
    unaugmented, _ = prepare_model_input(hypercubes[0], (0, 1, 2), mean, std, size=(8, 8))

    torch.manual_seed(1729)
    first = [dataset[index][0] for index in range(len(dataset))]
    torch.manual_seed(1729)
    again = [dataset[index][0] for index in range(len(dataset))]
    torch.manual_seed(2718)
    elsewhere = [dataset[index][0] for index in range(len(dataset))]

    assert all(torch.equal(left, right) for left, right in zip(first, again, strict=True))
    # Not vacuous: the draws really do augment, and another seed draws otherwise.
    assert not torch.equal(first[0], unaugmented)
    assert not all(torch.equal(left, right) for left, right in zip(first, elsewhere, strict=True))


def test_augmentation_is_off_when_the_dataset_is_not_training(tmp_path):
    hypercubes, (mean, std) = augmentable(tmp_path)
    dataset = SelectedBandDataset(hypercubes, (0, 1, 2), mean, std, (8, 8), training=False)
    expected, _ = prepare_model_input(hypercubes[0], (0, 1, 2), mean, std, size=(8, 8))

    torch.manual_seed(0)
    image, label = dataset[0]

    assert torch.equal(image, expected)
    assert label == 0


# --- Data identity, candidate seed and metrics ------------------------------
# The seed of a candidate is derived from the candidate and from the data it
# will train on, so a fitness does not depend on when, or by whom, the
# candidate was found.  These tests watch that derivation, not its arithmetic.

WAVELENGTHS = [400.0, 500.0, 600.0, 700.0, 800.0]


def cohorts_of(tiny_dataset):
    """The tiny dataset's train-fit crops, validation crops and train rows."""
    manifest, rows = tiny_dataset
    return (
        load_hypercubes(manifest, rows["fit"], verbose=False),
        load_hypercubes(manifest, rows["validation"], verbose=False),
        rows["fit"] + rows["validation"],
    )


def identity_of(tiny_dataset, *, wavelengths=None):
    """The data identity of the tiny dataset's train-fit and validation crops."""
    training, validation, train_rows = cohorts_of(tiny_dataset)
    return data_identity(
        training,
        validation,
        train_rows,
        WAVELENGTHS if wavelengths is None else wavelengths,
    )


def test_candidate_seed_is_the_same_every_time(tiny_dataset):
    identity = identity_of(tiny_dataset)

    first = candidate_seed((0, 1, 2), identity)
    again = candidate_seed((0, 1, 2), identity)
    from_a_fresh_identity = candidate_seed((0, 1, 2), identity_of(tiny_dataset))

    assert first == again == from_a_fresh_identity
    assert 0 <= first < 2**32


def test_candidate_seed_differs_for_a_permutation_of_the_same_bands(tiny_dataset):
    identity = identity_of(tiny_dataset)

    seeds = {order: candidate_seed(order, identity) for order in ((0, 1, 2), (2, 1, 0), (1, 0, 2))}

    assert len(set(seeds.values())) == 3


def test_candidate_seed_differs_for_another_data_identity(tiny_dataset):
    identity = identity_of(tiny_dataset)
    # The same crops cut against another spectral axis are not the same data.
    elsewhere = identity_of(tiny_dataset, wavelengths=[401.0, *WAVELENGTHS[1:]])

    assert identity != elsewhere
    assert candidate_seed((0, 1, 2), identity) != candidate_seed((0, 1, 2), elsewhere)


def test_candidate_seed_follows_the_seed_it_is_given(tiny_dataset):
    identity = identity_of(tiny_dataset)

    assert candidate_seed((0, 1, 2), identity, seed=config.CANDIDATE_SEED) != candidate_seed(
        (0, 1, 2), identity, seed=config.CANDIDATE_SEED + 1
    )


def test_data_identity_ignores_the_order_the_partition_rows_arrive_in(tiny_dataset):
    training, validation, train_rows = cohorts_of(tiny_dataset)

    forwards = data_identity(training, validation, train_rows, WAVELENGTHS)
    backwards = data_identity(
        training[::-1], validation, list(reversed(train_rows)), WAVELENGTHS
    )

    assert forwards == backwards


def test_data_identity_separates_the_crops_it_fits_from_the_crops_it_scores(tiny_dataset):
    fit, validation, train_rows = cohorts_of(tiny_dataset)

    identity = data_identity(fit, validation, train_rows, WAVELENGTHS)
    swapped = data_identity(fit[:1], [*fit[1:], *validation], train_rows, WAVELENGTHS)

    assert identity["train_cropped_hypercubes"] != swapped["train_cropped_hypercubes"]
    assert identity["validation_cropped_hypercubes"] != swapped["validation_cropped_hypercubes"]
    # The partition rows and the axis did not move, so their checksums must not.
    assert identity["train_evaluation_assignments"] == swapped["train_evaluation_assignments"]
    assert identity["spectral_axis"] == swapped["spectral_axis"]


def test_data_identity_sees_a_changed_reflectance_value(tiny_dataset):
    training, validation, train_rows = cohorts_of(tiny_dataset)
    identity = data_identity(training, validation, train_rows, WAVELENGTHS)

    training[0].reflectance[0, 0, 0] = 0.123
    tampered = data_identity(training, validation, train_rows, WAVELENGTHS)

    assert identity["train_cropped_hypercubes"] != tampered["train_cropped_hypercubes"]


def test_metrics_score_every_class_even_when_one_never_appears():
    # C3 is in neither the truth nor the predictions; with explicit labels it
    # still gets a row, an F1 of zero rather than a nan, and a 4x4 matrix.
    metrics = classification_metrics([0, 0, 1, 2], [0, 1, 1, 2])

    assert metrics["per_class_recall"] == {
        "C0": 0.5,
        "C1": 1.0,
        "C2": 1.0,
        "C3": 0.0,
    }
    assert np.asarray(metrics["confusion_matrix"]).shape == (4, 4)
    assert metrics["ordinal_mae"] == 0.25
    assert all(
        np.isfinite(metrics[name])
        for name in ("weighted_f1", "macro_f1", "ordinal_mae", "quadratic_weighted_kappa")
    )


def test_metrics_are_perfect_on_a_perfect_prediction():
    metrics = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3])

    assert metrics["weighted_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["ordinal_mae"] == 0.0
    assert metrics["quadratic_weighted_kappa"] == 1.0


# --- The genetic search, its cache and its resume ---------------------------
# The search is exercised through an injected evaluator, so none of this needs
# a GPU or a hypercube.  What it watches is the trajectory: which candidates are
# proposed, in which order, and which of them cost a training run.

GA_IDENTITY = {
    "train_cropped_hypercubes": "a" * 64,
    "validation_cropped_hypercubes": "b" * 64,
    "train_evaluation_assignments": "c" * 64,
    "spectral_axis": "d" * 64,
}
GA_BANDS = 40
GA_WAVELENGTHS = [400.0 + index for index in range(GA_BANDS)]
GA_PARAMS = ga.SearchParameters(population_size=6, generations=4)


def fake_fitness(candidate):
    """A fitness that depends on the candidate alone, so a replay can be checked."""
    score = ((candidate[0] * 7 + candidate[1] * 3 + candidate[2]) % 97) / 97
    return ga.CandidateFitness(score, score * 0.9, 1 - score, score * 0.8)


def make_cache(directory, *, contract=None, params=None, identity=None):
    """A cache in `directory`, with the sidecar the experiment would have written."""
    return ga.CandidateCache(
        directory / "candidates.csv",
        directory / "config.json",
        identity=GA_IDENTITY if identity is None else identity,
        wavelengths=GA_WAVELENGTHS,
        fitness_contract={"epochs": 1} if contract is None else contract,
        ga_parameters=(GA_PARAMS if params is None else params).as_dict(),
    )


def run_ga(directory, evaluator, **kwargs):
    """One search over a warm or cold cache in `directory`."""
    cache = make_cache(directory, **kwargs)
    random.seed(config.STUDY_SEED)
    return ga.run_search(GA_BANDS, evaluator, cache, params=GA_PARAMS), cache


def test_ga_operators_preserve_order_distinctness_and_axis_bounds():
    rng = random.Random(19)

    initialized = [ga.initialize_candidate(rng, 5) for _ in range(20)]
    left, right = ga.crossover_candidates((4, 0, 1), (0, 4, 3), rng, band_count=5, alpha=0.5)
    mutated = ga.mutate_candidate(
        (4, 0, 1), rng, band_count=5, mu=0.0, sigma=100.0, gene_probability=1.0
    )
    repaired = ga.repair_candidate((3, 3, 1), rng, band_count=5)

    assert all(len(set(candidate)) == 3 for candidate in [*initialized, left, right, mutated])
    assert all(
        all(0 <= index < 5 for index in candidate)
        for candidate in [*initialized, left, right, mutated]
    )
    # The repair replaces the duplicate in place: the bands that were already
    # distinct do not move, so a candidate is never silently reordered.
    assert repaired[0] == 3
    assert repaired[2] == 1
    assert repaired[1] not in {3, 1}
    with pytest.raises(ValueError, match="at least three bands"):
        ga.initialize_candidate(rng, 2)


def test_ga_search_evaluates_each_candidate_once_and_repeats_from_cache(tmp_path):
    seen = []
    result, cache = run_ga(tmp_path, lambda candidate: (seen.append(candidate), fake_fitness(candidate))[1])

    assert seen == list(dict.fromkeys(seen))  # never the same candidate twice
    assert len(seen) == len(result.fitness) == len(cache.entries)
    # Not vacuous: the population really does revisit candidates it has seen.
    assert result.cache_hits > 0


def test_ga_search_keeps_a_permutation_of_the_same_bands_apart(tmp_path):
    cache = make_cache(tmp_path)
    cache.append((1, 2, 3), fake_fitness((1, 2, 3)), first_generation=0, seconds=1.0)
    cache.append((3, 2, 1), fake_fitness((3, 2, 1)), first_generation=0, seconds=1.0)

    assert cache.get((1, 2, 3)) != cache.get((3, 2, 1))
    assert make_cache(tmp_path).restored == 2


def test_ga_search_is_not_disturbed_by_an_evaluator_that_reseeds_random(tmp_path):
    quiet = []
    noisy = []

    def reseeds_random(candidate):
        noisy.append(candidate)
        random.seed(sum(candidate))
        return fake_fitness(candidate)

    calm, _ = run_ga(tmp_path / "calm", lambda c: (quiet.append(c), fake_fitness(c))[1])
    loud, _ = run_ga(tmp_path / "loud", reseeds_random)

    assert quiet == noisy
    assert calm.winner == loud.winner


def test_ga_search_resumes_from_a_warm_cache_without_evaluating(tmp_path):
    first, _ = run_ga(tmp_path, lambda c: fake_fitness(c))

    calls = []
    second, cache = run_ga(tmp_path, lambda c: calls.append(c))

    assert cache.restored == len(first.fitness)
    assert calls == []
    assert second.winner == first.winner
    assert second.generations == first.generations


def test_ga_search_without_evaluation_refuses_a_cache_miss(tmp_path):
    with pytest.raises(LookupError, match="not in the cache"):
        run_ga(tmp_path, ga.refuse_to_evaluate)


def test_ga_cache_refuses_a_row_written_under_another_candidate_seed(tmp_path):
    cache = make_cache(tmp_path)
    cache.append((1, 2, 3), fake_fitness((1, 2, 3)), first_generation=0, seconds=1.0)
    rows = (tmp_path / "candidates.csv").read_text().splitlines()
    rows[1] = rows[1].replace(str(ga.candidate_seed((1, 2, 3), GA_IDENTITY)), "12345")
    (tmp_path / "candidates.csv").write_text("\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="candidate seed"):
        make_cache(tmp_path)


def test_ga_cache_refuses_a_different_fitness_contract(tmp_path):
    make_cache(tmp_path, contract={"epochs": 1})

    with pytest.raises(ValueError, match="fitness contract"):
        make_cache(tmp_path, contract={"epochs": 2})


def test_ga_cache_only_warns_when_the_ga_parameters_differ(tmp_path, capsys):
    make_cache(tmp_path)

    make_cache(tmp_path, params=ga.SearchParameters(population_size=8, generations=4))

    assert "GA parameters" in capsys.readouterr().out


def test_ga_winner_is_re_tie_broken_over_every_evaluated_candidate(tmp_path):
    # Every candidate ties on weighted F1, so only the tie-break decides, and it
    # must reach candidates DEAP's hall of fame never held.
    tied = ga.CandidateFitness(0.5, 0.5, 0.25, 0.5)
    result, cache = run_ga(tmp_path, lambda candidate: tied)

    assert result.winner == max(cache.entries, key=lambda c: tuple(-index for index in c))


def test_ga_winner_is_never_a_candidate_the_search_did_not_propose(tmp_path):
    # The cache can hold rows this trajectory never visits: a run with other GA
    # parameters left them, or ticket 06 preloaded them.  A winner has to be a
    # candidate the search actually met, however good a stranger's row looks.
    tied = ga.CandidateFitness(0.5, 0.5, 0.25, 0.5)
    cache = make_cache(tmp_path)
    stranger = (0, 1, 2)  # ties, and its low indices would win the tie-break
    cache.append(stranger, tied, first_generation=0, seconds=1.0)

    random.seed(config.STUDY_SEED)
    result = ga.run_search(GA_BANDS, lambda candidate: tied, cache, params=GA_PARAMS)

    assert stranger not in result.fitness
    assert result.winner != stranger
    assert result.winner in result.fitness


# --- The final model and the single read of the test set --------------------


FINAL_WAVELENGTHS = [400.0, 450.0, 500.0, 550.0, 600.0]  # one per band of `crop`
FINAL_BANDS = (3, 0, 2)


def write_spectral_axes(directory, wavelengths):
    """A SpectralAxis manifest whose id is the content address of its wavelengths."""
    path = directory / "spectral_axes.csv"
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=SPECTRAL_AXIS_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(
            {
                "schema_version": "1",
                "spectral_axis_id": spectral_axis_id(wavelengths),
                "band_count": str(len(wavelengths)),
                "wavelengths_nm": json.dumps(wavelengths, separators=(",", ":")),
            }
        )
    return path


def write_partitions(directory, rows):
    path = directory / "evaluation_partitions.csv"
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=PARTITION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def final_dataset(tmp_path, monkeypatch):
    """A whole dataset — train-fit, validation and held-out test — config pointed at it.

    The two test crops are in the manifest in an order that is neither sorted
    nor the order the partition rows name them in, so an evaluation that lost
    the manifest order would show up in the predictions file.
    """
    axis = write_spectral_axes(tmp_path, FINAL_WAVELENGTHS)
    crops = [
        ("fit-a", "acq-fit", "C0", crop(0.2)),
        ("fit-b", "acq-fit-two", "C1", crop(0.5)),
        ("val-a", "acq-validation", "C2", crop(0.4)),
        ("test-z", "acq-test", "C3", crop(0.8)),
        ("test-a", "acq-test", "C3", crop(0.7)),
    ]
    manifest = write_dataset(tmp_path, crops, axis_id=spectral_axis_id(FINAL_WAVELENGTHS))
    partitions = write_partitions(
        tmp_path,
        [
            partition_row("fit-a", "acq-fit", "C0", "train", "train"),
            partition_row("fit-b", "acq-fit-two", "C1", "train", "train"),
            partition_row("val-a", "acq-validation", "C2", "train", "validation"),
            partition_row("test-a", "acq-test", "C3", "test", ""),
            partition_row("test-z", "acq-test", "C3", "test", ""),
        ],
    )
    for name, value in (
        ("HYPERCUBES_MANIFEST", manifest),
        ("SPECTRAL_AXES", axis),
        ("PARTITIONS", partitions),
        ("TABLES_DIR", tmp_path / "tables"),
        ("FIGURES_DIR", tmp_path / "figures"),
        ("MODELS_DIR", tmp_path / "models"),
        ("DEVICE", "cpu"),
    ):
        monkeypatch.setattr(config, name, value)
    return tmp_path


def fake_final_model(epochs=2):
    """What an injected trainer hands back: a state dict and one row per epoch."""
    return train_final.FinalModel(
        state={"fc.bias": torch.zeros(config.NUM_CLASSES)},
        history=[
            {"epoch": epoch, "train_loss": 1.0 / epoch, "train_accuracy": 0.25 * epoch}
            for epoch in range(1, epochs + 1)
        ],
    )


def run_final(experiment=21, bands=FINAL_BANDS, *, trainer=None, evaluator=None, epochs=2):
    """One final-model run whose training and inference are replaced by fakes."""
    return train_final.final_command(
        experiment,
        bands=bands,
        epochs=epochs,
        trainer=trainer if trainer is not None else lambda context: fake_final_model(epochs),
        evaluator=evaluator if evaluator is not None else (lambda context: [0, 3]),
        verbose=False,
    )


def test_final_trains_on_every_train_crop_and_never_sees_a_test_crop(final_dataset):
    trained = []

    run_final(trainer=lambda context: (trained.append(context), fake_final_model())[1])

    (context,) = trained
    # Fit and validation together: the inner split did its job during the search.
    assert [item.cropped_hypercube_id for item in context.training] == ["fit-a", "fit-b", "val-a"]
    assert context.seed == config.FINAL_SEED
    # There is no test crop reachable from what the trainer is handed.
    assert not hasattr(context, "test")


def test_final_saves_the_model_before_the_test_set_is_opened(final_dataset, monkeypatch):
    """The claim of the whole script: no held-out crop is read until the model is frozen."""
    model_path = train_final.final_paths(21, FINAL_BANDS)["model"]
    opened = []
    load_crops = train_final.load_partition_crops

    def watched(manifest_path, partition, wavelengths, *, verbose):
        opened.append((partition, model_path.exists()))
        return load_crops(manifest_path, partition, wavelengths, verbose=verbose)

    monkeypatch.setattr(train_final, "load_partition_crops", watched)
    evaluated = []

    run_final(evaluator=lambda context: (evaluated.append(model_path.exists()), [0, 3])[1])

    assert opened == [("train", False), ("test", True)]
    assert evaluated == [True]
    assert torch.load(model_path, weights_only=True)["fc.bias"].tolist() == [0.0] * 4


def test_final_evaluates_the_held_out_crops_once_in_manifest_order(final_dataset):
    evaluated = []

    run_final(evaluator=lambda context: (evaluated.append(context), [0, 3])[1])

    (context,) = evaluated
    assert [item.cropped_hypercube_id for item in context.test] == ["test-z", "test-a"]
    assert context.state["fc.bias"].tolist() == [0.0] * 4
    rows = list(csv.DictReader(train_final.final_paths(21, FINAL_BANDS)["predictions"].open()))
    assert [row["cropped_hypercube_id"] for row in rows] == ["test-z", "test-a"]
    assert [row["actual"] for row in rows] == ["C3", "C3"]
    assert [row["predicted"] for row in rows] == ["C0", "C3"]
    assert [row["acquisition_id"] for row in rows] == ["acq-test", "acq-test"]


def test_final_writes_its_tables_and_figures_with_the_bands_in_their_names(final_dataset):
    run_final()

    paths = train_final.final_paths(21, FINAL_BANDS)
    assert all(path.exists() for path in paths.values())
    assert {path.name for path in paths.values()} == {
        "exp_21_model_3_0_2.pt",
        "exp_21_final_metrics_3_0_2.json",
        "exp_21_confusion_matrix_3_0_2.csv",
        "exp_21_confusion_matrix_3_0_2.png",
        "exp_21_test_predictions_3_0_2.csv",
        "exp_21_training_history_3_0_2.csv",
        "exp_21_training_history_3_0_2.png",
    }
    metrics = json.loads(paths["metrics"].read_text())
    assert metrics["selected_band_indices"] == [3, 0, 2]
    assert metrics["selected_wavelengths_nm"] == [550.0, 400.0, 500.0]
    assert metrics["seed"] == config.FINAL_SEED
    assert metrics["test_evaluation_count"] == 1
    assert metrics["train_cropped_hypercube_count"] == 3
    assert metrics["test_cropped_hypercube_count"] == 2
    assert set(metrics["metrics"]["per_class_recall"]) == set(config.CLASS_NAMES)
    assert len(metrics["normalization"]["mean"]) == 3
    history = list(csv.DictReader(paths["history"].open()))
    assert [row["epoch"] for row in history] == ["1", "2"]
    assert tuple(history[0]) == ("epoch", "train_loss", "train_accuracy")
    confusion = list(csv.reader(paths["confusion"].open()))
    assert confusion[0] == ["actual", *(f"predicted_{name}" for name in config.CLASS_NAMES)]
    assert [row[0] for row in confusion[1:]] == list(config.CLASS_NAMES)
    # One of the two C3 crops was predicted C0 and the other C3.
    assert confusion[4] == ["C3", "1", "0", "0", "1"]


def test_final_reads_the_winner_from_the_ga_summary_when_no_bands_are_given(final_dataset):
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (config.TABLES_DIR / "exp_21_ga_summary.json").write_text(
        json.dumps({"winner": {"selected_band_indices": [4, 1, 0]}})
    )
    trained = []

    run_final(bands=None, trainer=lambda context: (trained.append(context), fake_final_model())[1])

    assert trained[0].bands == (4, 1, 0)
    assert train_final.final_paths(21, (4, 1, 0))["metrics"].exists()


def test_final_without_bands_or_a_summary_says_which_file_is_missing(final_dataset):
    with pytest.raises(ValueError, match="exp_21_ga_summary.json"):
        run_final(bands=None)


def test_final_refuses_an_experiment_number_an_earlier_protocol_owns(final_dataset):
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (config.TABLES_DIR / "exp_07_ga_stats.csv").write_text("gen\n0\n")

    with pytest.raises(ValueError, match="earlier run|new experiment number"):
        run_final(experiment=7)


def test_final_writes_a_free_experiment_number_and_its_own_number_again(final_dataset):
    run_final(experiment=99)
    first = train_final.final_paths(99, FINAL_BANDS)["metrics"].read_text()

    run_final(experiment=99)

    assert train_final.final_paths(99, FINAL_BANDS)["metrics"].read_text() == first


def test_final_refuses_a_rerun_that_would_not_reproduce_the_recorded_result(final_dataset):
    """A replay rewrites the same bytes; another epoch count would overwrite a result."""
    run_final(experiment=99, epochs=2)
    recorded = train_final.final_paths(99, FINAL_BANDS)["metrics"].read_text()

    with pytest.raises(ValueError, match="different epochs"):
        run_final(experiment=99, epochs=1)

    assert train_final.final_paths(99, FINAL_BANDS)["metrics"].read_text() == recorded


# --- The grouped bootstrap interval -----------------------------------------


BOOTSTRAP_BANDS = (3, 0, 2)


def write_test_predictions(experiment, bands, rows):
    """A `train_final.py` predictions file: one row per held-out crop."""
    path = train_final.final_paths(experiment, bands)["predictions"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("cropped_hypercube_id", "acquisition_id", "actual", "predicted"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_final_metrics(experiment, bands, nanometres=(550.0, 400.0, 500.0)):
    """The part of `exp_NN_final_metrics_*.json` the bootstrap reads."""
    path = train_final.final_paths(experiment, bands)["metrics"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "selected_band_indices": list(bands),
                "selected_wavelengths_nm": list(nanometres),
            }
        )
    )
    return path


def prediction_rows(*specs):
    """`("acq", "C0", "C1")` is one crop of `acq` whose C0 was called C1."""
    return [
        {
            "cropped_hypercube_id": f"crop-{number}",
            "acquisition_id": acquisition,
            "actual": actual,
            "predicted": predicted,
        }
        for number, (acquisition, actual, predicted) in enumerate(specs)
    ]


BOOTSTRAP_ROWS = prediction_rows(
    ("acq-right", "C0", "C0"),
    ("acq-right", "C0", "C0"),
    ("acq-right", "C1", "C1"),
    ("acq-wrong", "C2", "C3"),
    ("acq-wrong", "C2", "C3"),
    ("acq-half", "C3", "C3"),
    ("acq-half", "C3", "C0"),
)


@pytest.fixture
def bootstrap_dataset(tmp_path, monkeypatch):
    """One published final model of experiment 21, and nothing else to read.

    `HYPERCUBES_MANIFEST` and `MODELS_DIR` are pointed at paths that do not
    exist, so a bootstrap that opened a crop or a checkpoint would fail here.
    """
    for name, value in (
        ("TABLES_DIR", tmp_path / "tables"),
        ("FIGURES_DIR", tmp_path / "figures"),
        ("MODELS_DIR", tmp_path / "no-models"),
        ("HYPERCUBES_MANIFEST", tmp_path / "no-manifest.csv"),
        ("SPECTRAL_AXES", tmp_path / "no-axes.csv"),
        ("PARTITIONS", tmp_path / "no-partitions.csv"),
    ):
        monkeypatch.setattr(config, name, value)
    write_test_predictions(21, BOOTSTRAP_BANDS, BOOTSTRAP_ROWS)
    write_final_metrics(21, BOOTSTRAP_BANDS)
    return tmp_path


def run_bootstrap(experiment=21, *, bands=None, resamples=32, seed=7):
    return bootstrap.bootstrap_command(
        experiment, bands=bands, resamples=resamples, seed=seed, verbose=False
    )


def test_bootstrap_resamples_whole_acquisitions_and_never_a_single_crop():
    acquisition = np.array(["a", "a", "a", "b", "c", "c"])
    members = {name: np.flatnonzero(acquisition == name) for name in ("a", "b", "c")}

    draws = bootstrap.grouped_resamples(acquisition, resamples=50, seed=3)

    assert len(draws) == 50
    for index in draws:
        counts = {
            name: int(np.count_nonzero(np.isin(index, member)))
            for name, member in members.items()
        }
        # Every crop of a drawn acquisition comes along, as many times as it was
        # drawn, and the three draws add up to the length of the resample.
        for name, member in members.items():
            assert counts[name] % len(member) == 0
        assert sum(counts.values()) == len(index)
        assert sum(counts[name] // len(members[name]) for name in members) == len(members)


def test_bootstrap_repeats_exactly_under_the_same_seed_and_moves_under_another():
    acquisition = np.array(["a", "a", "b", "c"])

    first = bootstrap.grouped_resamples(acquisition, resamples=20, seed=11)
    again = bootstrap.grouped_resamples(acquisition, resamples=20, seed=11)
    other = bootstrap.grouped_resamples(acquisition, resamples=20, seed=12)

    assert all(np.array_equal(one, two) for one, two in zip(first, again, strict=True))
    assert not all(np.array_equal(one, two) for one, two in zip(first, other, strict=True))


def bootstrap_predictions(rows=None):
    """`BOOTSTRAP_ROWS` as the three aligned columns the bootstrap works on."""
    rows = BOOTSTRAP_ROWS if rows is None else rows
    return bootstrap.TestPredictions(
        *(
            np.array([row[column] for row in rows])
            for column in ("actual", "predicted", "acquisition_id")
        )
    )


def test_bootstrap_interval_is_the_percentile_pair_of_the_resampled_scores():
    predictions = bootstrap_predictions()

    interval = bootstrap.bootstrap_interval(predictions, resamples=64, seed=5)

    assert interval.point == bootstrap.weighted_f1(predictions)
    assert len(interval.scores) == 64
    assert (interval.low, interval.high) == tuple(
        np.percentile(interval.scores, list(config.BOOTSTRAP_PERCENTILES))
    )
    assert interval.low <= interval.high


def test_bootstrap_per_acquisition_table_names_every_acquisition_worst_first():
    table = bootstrap.per_acquisition_scores(bootstrap_predictions())

    assert [row["acquisition_id"] for row in table] == ["acq-wrong", "acq-half", "acq-right"]
    assert [row["cropped_hypercube_count"] for row in table] == [2, 2, 3]
    assert [row["majority_class"] for row in table] == ["C2", "C3", "C0"]
    assert table[0]["weighted_f1"] == 0.0 and table[0]["accuracy"] == 0.0
    assert table[2]["weighted_f1"] == 1.0 and table[2]["accuracy"] == 1.0
    assert [row["weighted_f1"] for row in table] == sorted(row["weighted_f1"] for row in table)


def test_bootstrap_reads_the_only_final_model_of_the_experiment(bootstrap_dataset):
    """No `--bands`: one published triplet is not ambiguous."""
    run_bootstrap()

    written = json.loads(bootstrap.bootstrap_paths(21, BOOTSTRAP_BANDS)["bootstrap"].read_text())
    assert written["selected_band_indices"] == list(BOOTSTRAP_BANDS)
    assert written["selected_wavelengths_nm"] == [550.0, 400.0, 500.0]
    assert written["resamples"] == 32 and written["seed"] == 7
    assert written["test_cropped_hypercube_count"] == 7
    assert written["test_acquisition_count"] == 3
    assert len(written["weighted_f1_ci95"]) == 2
    assert len(written["per_acquisition"]) == 3


def test_bootstrap_says_which_experiment_has_no_final_model(bootstrap_dataset):
    with pytest.raises(ValueError, match="no final model|exp_22"):
        run_bootstrap(experiment=22)


def test_bootstrap_asks_for_bands_when_the_experiment_has_two_final_models(bootstrap_dataset):
    other = (1, 2, 4)
    write_test_predictions(21, other, BOOTSTRAP_ROWS)
    write_final_metrics(21, other)

    with pytest.raises(ValueError, match="--bands"):
        run_bootstrap()

    run_bootstrap(bands=other)
    assert bootstrap.bootstrap_paths(21, other)["bootstrap"].exists()


def test_bootstrap_of_a_triplet_with_no_predictions_names_the_missing_file(bootstrap_dataset):
    train_final.final_paths(21, BOOTSTRAP_BANDS)["predictions"].unlink()

    with pytest.raises(ValueError, match="exp_21_test_predictions_3_0_2.csv"):
        run_bootstrap()


def test_bootstrap_run_twice_writes_the_same_json(bootstrap_dataset):
    run_bootstrap()
    first = bootstrap.bootstrap_paths(21, BOOTSTRAP_BANDS)["bootstrap"].read_text()

    run_bootstrap()

    assert bootstrap.bootstrap_paths(21, BOOTSTRAP_BANDS)["bootstrap"].read_text() == first


def test_bootstrap_refuses_a_rerun_that_would_not_reproduce_the_recorded_interval(
    bootstrap_dataset,
):
    run_bootstrap(resamples=32, seed=7)
    recorded = bootstrap.bootstrap_paths(21, BOOTSTRAP_BANDS)["bootstrap"].read_text()

    with pytest.raises(ValueError, match="different resamples"):
        run_bootstrap(resamples=64, seed=7)
    with pytest.raises(ValueError, match="different seed"):
        run_bootstrap(resamples=32, seed=8)

    assert bootstrap.bootstrap_paths(21, BOOTSTRAP_BANDS)["bootstrap"].read_text() == recorded


def test_bootstrap_refuses_an_interval_whose_predictions_have_changed_underneath(
    bootstrap_dataset,
):
    """Same bands, same seed, same resamples, other predictions: another result."""
    run_bootstrap()
    recorded = bootstrap.bootstrap_paths(21, BOOTSTRAP_BANDS)["bootstrap"].read_text()
    write_test_predictions(
        21,
        BOOTSTRAP_BANDS,
        prediction_rows(*[("acq-right", "C0", "C0")] * 3, *[("acq-wrong", "C2", "C2")] * 4),
    )

    with pytest.raises(ValueError, match="different weighted_f1"):
        run_bootstrap()

    assert bootstrap.bootstrap_paths(21, BOOTSTRAP_BANDS)["bootstrap"].read_text() == recorded


RECORDED_PREDICTIONS = (
    config.ROOT / "out" / "tables" / "exp_21_test_predictions_366_262_225.csv"
)


@pytest.mark.skipif(
    not RECORDED_PREDICTIONS.exists(), reason="experiment 21 has not been run here"
)
def test_bootstrap_reproduces_the_recorded_interval_of_experiment_21():
    """The number the thesis quotes: 0.722, 95% CI [0.645, 0.848]."""
    predictions = bootstrap.read_predictions(RECORDED_PREDICTIONS)

    interval = bootstrap.bootstrap_interval(
        predictions,
        resamples=config.BOOTSTRAP_RESAMPLES,
        seed=config.BOOTSTRAP_SEED,
    )

    assert interval.point == 0.722142952443074
    assert (round(interval.low, 3), round(interval.high, 3)) == (0.645, 0.848)


# --- Redrawing the fitness curve of a finished search ------------------------


def generation_records(count=3):
    """What `run_search` hands to `write_stats`: one record per generation."""
    return [
        {
            "gen": gen,
            "nevals": 4 - gen,
            "population_size": 4,
            "unique_population_candidates": 4 - gen,
            "avg": 0.50 + gen / 100,
            "std": 0.01,
            "min": 0.40,
            "max": 0.60 + gen / 100,
            "best": (1, 2, 3),
        }
        for gen in range(count)
    ]


@pytest.fixture
def searched_experiment(tmp_path, monkeypatch):
    """The two tables a finished search leaves behind, and an empty figures dir."""
    for name, value in (
        ("TABLES_DIR", tmp_path / "tables"),
        ("FIGURES_DIR", tmp_path / "figures"),
    ):
        monkeypatch.setattr(config, name, value)
    paths = ga.experiment_paths(21)
    ga.write_stats(paths["stats"], generation_records(), FINAL_WAVELENGTHS)
    paths["summary"].write_text(
        json.dumps({"winner": {"selected_wavelengths_nm": [891.02, 747.5, 697.05]}})
    )
    return paths


def test_plot_reads_back_the_generations_the_search_wrote(searched_experiment):
    rows = plot_fitness_evolution.read_generations(searched_experiment["stats"])

    assert [row["gen"] for row in rows] == [0, 1, 2]
    assert [row["avg"] for row in rows] == [0.50, 0.51, 0.52]
    assert rows[0]["nevals"] == 4


def test_plot_refuses_the_stats_of_a_search_before_experiment_21(tmp_path):
    path = tmp_path / "exp_10_ga_stats.csv"
    path.write_text("gen,nevals,avg,std,min,max,best\n0,20,0.69,0.05,0.56,0.76,\"[3, 1, 2]\"\n")

    with pytest.raises(ValueError, match="21"):
        plot_fitness_evolution.read_generations(path)


def test_plot_takes_the_title_from_the_winner_of_the_summary(searched_experiment):
    assert plot_fitness_evolution.winner_wavelengths(searched_experiment["summary"]) == (
        891.02,
        747.5,
        697.05,
    )


def test_plot_redraws_the_figure_of_the_experiment_it_is_given(searched_experiment):
    figure = plot_fitness_evolution.plot_command(21, verbose=False)

    assert figure == searched_experiment["figure"]
    assert figure.name == "exp_21_fitness_evolution.png"
    assert figure.read_bytes().startswith(b"\x89PNG")


def test_plot_says_which_file_an_unsearched_experiment_is_missing(searched_experiment):
    with pytest.raises(ValueError, match="exp_22_ga_stats.csv"):
        plot_fitness_evolution.plot_command(22, verbose=False)

    assert plot_fitness_evolution.main(["22"]) == 1


# --- Verifying the linked crops once ----------------------------------------


SANITY_WAVELENGTHS = [400.0, 450.0, 500.0, 550.0, 600.0]  # one per band of `crop`
SANITY_BANDS = (4, 1, 2)


def sanity_argv(*extra):
    """The command line of a check whose bands are inside the tiny axis."""
    return [*extra, "--bands", *(str(band) for band in SANITY_BANDS)]


def rewrite_checksums(manifest):
    """Put each artifact's real SHA-256 in the manifest that names it."""
    rows = list(csv.DictReader(manifest.open(newline="")))
    for row in rows:
        artifact = manifest.parent / row["artifact_relative_path"]
        row["artifact_checksum"] = check_data.artifact_checksum(artifact)
    with manifest.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=CROPPED_HYPERCUBE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


@pytest.fixture
def sanity_dataset(tmp_path, monkeypatch):
    """Three train crops of every class and one held-out crop, checksums recorded."""
    axis = write_spectral_axes(tmp_path, SANITY_WAVELENGTHS)
    crops = [
        (
            f"{severity.lower()}-{index}",
            f"acq-{severity.lower()}-{'val' if index == 2 else 'fit'}",
            severity,
            crop(value),
        )
        for severity, values in (
            ("C0", (0.11, 0.12, 0.13)),
            ("C1", (0.21, 0.22, 0.23)),
            ("C2", (0.31, 0.32, 0.33)),
            ("C3", (0.41, 0.42, 0.43)),
        )
        for index, value in enumerate(values)
    ]
    crops.append(("test-a", "acq-test", "C3", crop(0.9)))
    manifest = write_dataset(tmp_path, crops, axis_id=spectral_axis_id(SANITY_WAVELENGTHS))
    rewrite_checksums(manifest)
    partitions = write_partitions(
        tmp_path,
        [
            partition_row(
                identity,
                acquisition,
                severity,
                "train",
                "validation" if acquisition.endswith("-val") else "train",
            )
            for identity, acquisition, severity, _ in crops[:-1]
        ]
        + [partition_row("test-a", "acq-test", "C3", "test", "")],
    )
    for name, value in (
        ("HYPERCUBES_MANIFEST", manifest),
        ("SPECTRAL_AXES", axis),
        ("PARTITIONS", partitions),
        ("SANITY_CHECK_DIR", tmp_path / "sanity-check"),
    ):
        monkeypatch.setattr(config, name, value)
    return tmp_path


def corrupt(manifest, identity):
    """Overwrite one artifact with a different, still valid, crop."""
    reflectance, mask = crop(0.55)
    np.savez_compressed(
        manifest.parent / "cropped_hypercubes" / f"{identity}.npz",
        reflectance=reflectance,
        foreground_mask=mask,
    )


def test_check_data_reports_no_mismatch_when_every_artifact_hashes_as_recorded(sanity_dataset):
    manifest = config.HYPERCUBES_MANIFEST

    assert check_data.mismatched_artifacts(manifest, load_manifest(manifest)) == []


def test_check_data_names_the_crop_whose_artifact_changed(sanity_dataset):
    manifest = config.HYPERCUBES_MANIFEST
    corrupt(manifest, "c2-1")

    assert check_data.mismatched_artifacts(manifest, load_manifest(manifest)) == ["c2-1"]


def test_check_data_counts_an_artifact_that_is_not_there_as_a_mismatch(sanity_dataset):
    manifest = config.HYPERCUBES_MANIFEST
    (manifest.parent / "cropped_hypercubes" / "c0-0.npz").unlink()

    assert check_data.mismatched_artifacts(manifest, load_manifest(manifest)) == ["c0-0"]


def test_check_data_draws_only_from_train_crops(sanity_dataset):
    crops = load_partition_crops(
        config.HYPERCUBES_MANIFEST, "train", SANITY_WAVELENGTHS, verbose=False
    )

    assert [hypercube.cropped_hypercube_id for hypercube in crops] == [
        f"{severity.lower()}-{index}"
        for severity in config.CLASS_NAMES
        for index in range(3)
    ]


def test_check_data_shows_the_same_number_of_crops_from_every_class(sanity_dataset):
    crops = load_partition_crops(
        config.HYPERCUBES_MANIFEST, "train", SANITY_WAVELENGTHS, verbose=False
    )

    shown = check_data.crops_to_show(crops, 2)

    assert [hypercube.severity_class for hypercube in shown] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_check_data_refuses_to_draw_a_class_it_has_too_few_crops_of(sanity_dataset):
    crops = load_partition_crops(
        config.HYPERCUBES_MANIFEST, "train", SANITY_WAVELENGTHS, verbose=False
    )

    with pytest.raises(ValueError, match="C0"):
        check_data.crops_to_show(crops[:1], 2)


def test_check_data_writes_the_grid_and_says_nothing_mismatched(sanity_dataset, capsys):
    exit_code = check_data.main(sanity_argv())

    assert exit_code == 0
    assert "13 files, 0 checksum mismatches" in capsys.readouterr().out
    assert (config.SANITY_CHECK_DIR / "data_sanity_check.png").exists()


def test_check_data_stops_before_the_figure_when_an_artifact_does_not_match(
    sanity_dataset, capsys
):
    corrupt(config.HYPERCUBES_MANIFEST, "c1-2")

    exit_code = check_data.main(sanity_argv())

    assert exit_code == 1
    assert "13 files, 1 checksum mismatch" in capsys.readouterr().out
    assert not (config.SANITY_CHECK_DIR / "data_sanity_check.png").exists()


def test_check_data_verifies_the_manifest_it_is_given(sanity_dataset, tmp_path, capsys):
    elsewhere = config.HYPERCUBES_MANIFEST.rename(tmp_path / "another_manifest.csv")

    exit_code = check_data.main(sanity_argv("--manifest", str(elsewhere)))

    assert exit_code == 0
    assert "13 files, 0 checksum mismatches" in capsys.readouterr().out


# --- The test errors of one experiment ---------------------------------------


ERROR_WAVELENGTHS = [400.0, 450.0, 500.0, 550.0, 600.0]  # one per band of `crop`
ERROR_BANDS = (4, 1, 2)

# identity, acquisition, actual, predicted, foreground reflectance, foreground pixels
ERROR_CROPS = (
    ("a-0", "acq-a", "C0", "C0", 0.5, 4),
    ("a-1", "acq-a", "C0", "C0", 0.5, 4),
    ("b-0", "acq-b", "C1", "C1", 0.5, 4),
    ("b-1", "acq-b", "C1", "C1", 0.5, 4),
    ("c-0", "acq-c", "C2", "C2", 0.5, 4),
    ("c-1", "acq-c", "C2", "C3", 0.5, 4),
    ("d-0", "acq-d", "C3", "C0", 0.1, 1),
    ("d-1", "acq-d", "C3", "C0", 0.1, 1),
)


def error_prediction_rows():
    """The fixture's crops as the rows `train_final.py` would have written."""
    return [
        {
            "cropped_hypercube_id": identity,
            "acquisition_id": acquisition,
            "actual": actual,
            "predicted": predicted,
        }
        for identity, acquisition, actual, predicted, _, _ in ERROR_CROPS
    ]


def error_crop(value, foreground_pixels):
    """A 2x2 crop whose first `foreground_pixels` pixels hold `value` per band."""
    mask = np.zeros((2, 2), dtype=np.bool_)
    mask.reshape(-1)[:foreground_pixels] = True
    return crop(value, height=2, width=2, foreground=mask)


@pytest.fixture
def errors_dataset(tmp_path, monkeypatch):
    """Four test acquisitions of one class each, two of them misread, plus train."""
    axis = write_spectral_axes(tmp_path, ERROR_WAVELENGTHS)
    held_out = [
        (identity, acquisition, actual, error_crop(value, pixels))
        for identity, acquisition, actual, _, value, pixels in ERROR_CROPS
    ]
    fitted = [
        ("train-0", "acq-train", "C0", error_crop(0.5, 4)),
        ("validation-0", "acq-validation", "C1", error_crop(0.5, 4)),
    ]
    manifest = write_dataset(
        tmp_path, held_out + fitted, axis_id=spectral_axis_id(ERROR_WAVELENGTHS)
    )
    partitions = write_partitions(
        tmp_path,
        [
            partition_row(identity, acquisition, severity, "test", "")
            for identity, acquisition, severity, _ in held_out
        ]
        + [
            partition_row("train-0", "acq-train", "C0", "train", "train"),
            partition_row("validation-0", "acq-validation", "C1", "train", "validation"),
        ],
    )
    for name, value in (
        ("TABLES_DIR", tmp_path / "tables"),
        ("FIGURES_DIR", tmp_path / "figures"),
        ("HYPERCUBES_MANIFEST", manifest),
        ("SPECTRAL_AXES", axis),
        ("PARTITIONS", partitions),
    ):
        monkeypatch.setattr(config, name, value)
    write_test_predictions(21, ERROR_BANDS, error_prediction_rows())
    write_final_metrics(21, ERROR_BANDS, nanometres=(600.0, 450.0, 500.0))
    return tmp_path


def run_analysis(experiment=21, *, bands=None):
    return analyze_test_errors.analyze_command(experiment, bands=bands, verbose=False)


def error_predictions():
    """The predictions of the fixture, without going through a file."""
    return bootstrap.TestPredictions(
        *(
            np.array(column)
            for column in zip(
                *((actual, predicted, acquisition) for _, acquisition, actual, predicted, _, _ in ERROR_CROPS),
                strict=True,
            )
        )
    )


def analysis_table():
    """The rows `analyze_test_errors.py` wrote, in the order it wrote them."""
    path = analyze_test_errors.analysis_paths(21)["analysis"]
    return list(csv.DictReader(path.open(newline="")))


def test_error_analysis_scores_each_acquisition_exactly_as_the_bootstrap_does(errors_dataset):
    run_analysis()

    written = analysis_table()
    scored = bootstrap.per_acquisition_scores(error_predictions())
    assert [row["acquisition_id"] for row in written] == [
        row["acquisition_id"] for row in scored
    ]
    assert [float(row["weighted_f1"]) for row in written] == [
        row["weighted_f1"] for row in scored
    ]


def test_error_analysis_counts_every_actual_predicted_pair_per_acquisition(errors_dataset):
    run_analysis()

    counted = {row["acquisition_id"]: row for row in analysis_table()}
    assert counted["acq-d"]["C3_as_C0"] == "2"
    assert counted["acq-d"]["C3_as_C3"] == "0"
    assert counted["acq-c"]["C2_as_C2"] == "1"
    assert counted["acq-c"]["C2_as_C3"] == "1"
    assert counted["acq-a"]["C0_as_C0"] == "2"
    assert [row["error_count"] for row in analysis_table()] == ["2", "1", "0", "0"]


def test_error_analysis_marks_the_two_weakest_acquisitions_and_no_others(errors_dataset):
    run_analysis()

    grouped = {row["acquisition_id"]: row["group"] for row in analysis_table()}
    assert grouped == {
        "acq-d": "weak",
        "acq-c": "weak",
        "acq-a": "reference",
        "acq-b": "reference",
    }


def test_error_analysis_measures_exposure_and_segmentation_from_the_crops(errors_dataset):
    run_analysis()

    measured = {row["acquisition_id"]: row for row in analysis_table()}
    assert float(measured["acq-d"]["visible_reflectance_mean"]) == pytest.approx(0.1)
    assert float(measured["acq-a"]["visible_reflectance_mean"]) == pytest.approx(0.5)
    assert float(measured["acq-d"]["foreground_fraction"]) == pytest.approx(0.25)
    assert float(measured["acq-a"]["foreground_fraction"]) == pytest.approx(1.0)
    assert float(measured["acq-a"]["crop_area_px_mean"]) == pytest.approx(4.0)


def test_error_analysis_reports_only_the_band_ranges_the_spectral_axis_covers():
    covered = analyze_test_errors.ranges_in_axis(ERROR_WAVELENGTHS)

    assert [band_range.name for band_range in covered] == ["visible"]
    assert covered[0].indices == [0, 1, 2, 3, 4]


def test_error_analysis_draws_the_misclassified_crops_of_the_weak_acquisitions_only(
    errors_dataset,
):
    records = analyze_test_errors.read_records(
        train_final.final_paths(21, ERROR_BANDS)["predictions"]
    )

    shown = analyze_test_errors.misclassified(records, ("acq-d", "acq-c"))

    assert [entry.cropped_hypercube_id for entry in shown] == ["c-1", "d-0", "d-1"]
    assert [(entry.actual, entry.predicted) for entry in shown] == [
        ("C2", "C3"),
        ("C3", "C0"),
        ("C3", "C0"),
    ]


def test_error_analysis_writes_its_table_and_its_figure(errors_dataset):
    run_analysis()

    paths = analyze_test_errors.analysis_paths(21)
    assert paths["analysis"].name == "exp_21_test_error_analysis.csv"
    assert paths["figure"].name == "exp_21_test_errors.png"
    assert paths["analysis"].exists() and paths["figure"].exists()


def test_error_analysis_run_twice_writes_the_same_csv(errors_dataset):
    run_analysis()
    first = analyze_test_errors.analysis_paths(21)["analysis"].read_bytes()

    run_analysis()

    assert analyze_test_errors.analysis_paths(21)["analysis"].read_bytes() == first


def test_error_analysis_refuses_to_overwrite_the_analysis_of_another_triplet(errors_dataset):
    run_analysis()
    write_final_metrics(21, (0, 1, 2), nanometres=(400.0, 450.0, 500.0))
    write_test_predictions(21, (0, 1, 2), error_prediction_rows())

    with pytest.raises(ValueError, match="recorded for bands 4_1_2"):
        run_analysis(bands=(0, 1, 2))


@pytest.mark.skipif(
    not RECORDED_PREDICTIONS.exists(), reason="experiment 21 has not been run here"
)
def test_error_analysis_reproduces_the_seventeen_extreme_confusions_of_experiment_21():
    """The C0<->C3 block the ticket asks about, in the record the thesis quotes."""
    predictions = bootstrap.read_predictions(RECORDED_PREDICTIONS)

    counts = analyze_test_errors.confusion_counts(predictions)
    weak = analyze_test_errors.weak_acquisitions(predictions)

    assert counts[("C0", "C3")] + counts[("C3", "C0")] == 17
    assert [capture[:8] for capture in weak] == ["042ef12d", "f7ab3a6e"]
    assert counts[("C0", "C3")] == 13


def test_error_analysis_refuses_a_grid_with_no_misread_crop_to_draw():
    with pytest.raises(ValueError, match="no misread crops to draw"):
        analyze_test_errors.draw_errors([], {}, ERROR_BANDS, ERROR_WAVELENGTHS, Path("unused.png"))


def test_error_analysis_compares_the_crop_count_and_not_only_the_crops(errors_dataset):
    run_analysis()

    ranges = analyze_test_errors.ranges_in_axis(ERROR_WAVELENGTHS)
    fields = analyze_test_errors.comparison_fields(ranges)
    compared = analyze_test_errors.group_comparison(
        [{**row, **{field: float(row[field]) for field in fields}} for row in analysis_table()],
        fields,
    )

    assert compared[0]["measurement"] == "cropped_hypercube_count"
    assert compared[0]["weak"] == compared[0]["reference"] == 2.0


def test_error_analysis_says_where_the_extreme_confusions_fell():
    predictions = error_predictions()

    extreme = analyze_test_errors.extreme_confusions(predictions, ("acq-d",))

    assert extreme["pair"] == "C0<->C3"
    assert extreme["total"] == 2  # the two C3 crops of acq-d called C0
    assert extreme["in_weak"] == 2
    assert extreme["by_direction"] == {"C0->C3": 0, "C3->C0": 2}


def test_error_analysis_refuses_to_report_an_acquisition_it_could_not_measure():
    with pytest.raises(ValueError, match="no crop of acquisition acq-d"):
        analyze_test_errors.analysis_rows(error_predictions(), {}, ERROR_BANDS)
