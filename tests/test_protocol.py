"""What makes the result defensible, and nothing else.

These tests watch the protocol's observable behaviour: which crops each step
sees, which values come out, and what is refused.  They do not watch how any
of it is written.
"""

import csv
import json
import random

import numpy as np
import pytest
import torch

import config
import ga
from cnn.data_setup import (
    CROPPED_HYPERCUBE_FIELDS,
    PARTITION_FIELDS,
    SelectedBandDataset,
    apply_foreground_normalization,
    fit_foreground_normalization,
    load_hypercubes,
    load_manifest,
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


def write_dataset(directory, crops):
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
                "spectral_axis_id": "axis",
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
