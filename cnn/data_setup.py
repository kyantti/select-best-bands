"""The data path: manifests, partitions, hypercubes in RAM, and model input.

Ported from fig-aflatoxin's partition stage and evaluation module.  Every step
that trains or predicts goes through here, so the parts that make the result
defensible are copied literally: the partition checks that refuse a leak, the
manifest order the hypercubes are read in, normalization fitted on train
foreground pixels only, and augmentation that moves image and mask together.
"""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

import config

# --- Evaluation partitions -------------------------------------------------
# Ported from fig-aflatoxin's partition stage: the manifest schemas and the
# checks that refuse a manifest in which an Acquisition, or one of its crops,
# has ended up on both sides of a boundary.

CROPPED_HYPERCUBE_FIELDS = (
    "schema_version",
    "cropped_hypercube_id",
    "crop_id",
    "acquisition_id",
    "annotation_id",
    "severity_class",
    "height",
    "width",
    "band_count",
    "artifact_relative_path",
    "artifact_checksum",
    "spectral_axis_id",
)

PARTITION_FIELDS = (
    "schema_version",
    "cropped_hypercube_id",
    "crop_id",
    "acquisition_id",
    "severity_class",
    "partition",
    "selection_partition",
)


def read_manifest(path, fields: tuple[str, ...], *, name: str) -> list[dict[str, str]]:
    """Read a manifest CSV, refusing an unexpected schema or an empty file."""
    try:
        with open(path, newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != fields:
                raise ValueError(f"{name} has an incompatible schema")
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read {name} '{path}': {exc}") from exc
    if not rows:
        raise ValueError(f"{name} is empty")
    if any(row["schema_version"] != "1" for row in rows):
        raise ValueError(f"{name} has an unsupported schema version")
    return rows


def load_manifest(path) -> list[dict[str, str]]:
    """Load the CroppedHypercube manifest, checking identities and classes.

    Checksums are deliberately not verified on load; verifying the 1124 NPZ
    files once is `check_data.py`'s job, which ticket 10 gives it.
    """
    rows = read_manifest(path, CROPPED_HYPERCUBE_FIELDS, name="CroppedHypercube manifest")
    hypercube_ids = set()
    crop_ids = set()
    acquisition_classes = {}
    for row in rows:
        identity = row["cropped_hypercube_id"]
        crop_id = row["crop_id"]
        acquisition_id = row["acquisition_id"]
        if row["severity_class"] not in config.CLASS_NAMES:
            raise ValueError(f"invalid SeverityClass '{row['severity_class']}'")
        if not identity or identity in hypercube_ids:
            raise ValueError(f"duplicate or missing CroppedHypercube identity '{identity}'")
        if not crop_id or crop_id in crop_ids:
            raise ValueError(f"duplicate or missing crop identity '{crop_id}'")
        if not acquisition_id:
            raise ValueError("missing Acquisition identity")
        if PurePosixPath(identity).name != identity or "\\" in identity:
            raise ValueError(f"invalid CroppedHypercube identity '{identity}'")
        if any(
            not row[field]
            for field in ("annotation_id", "spectral_axis_id", "artifact_checksum")
        ):
            raise ValueError(f"missing lineage for CroppedHypercube '{identity}'")
        prior_class = acquisition_classes.setdefault(acquisition_id, row["severity_class"])
        if prior_class != row["severity_class"]:
            raise ValueError(f"conflicting SeverityClass for Acquisition '{acquisition_id}'")
        hypercube_ids.add(identity)
        crop_ids.add(crop_id)
    return rows


def validate_partition_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Refuse partition rows that leak across a boundary or lack a group.

    Raises:
        ValueError: If an Acquisition or one of its crops appears on both sides
            of the train/test or train/validation boundary, if a test record
            carries a selection partition, or if any of the three groups is
            missing.
    """
    hypercube_assignments = {}
    crop_assignments = {}
    acquisition_assignments = {}
    acquisition_classes = {}
    seen_hypercubes = set()
    seen_crops = set()
    assignments_seen = set()
    for row in rows:
        if row["severity_class"] not in config.CLASS_NAMES:
            raise ValueError(f"invalid SeverityClass '{row['severity_class']}'")
        partition = row["partition"]
        selection = row["selection_partition"]
        if partition == "test":
            if selection:
                raise ValueError("held-out test records cannot have a selection partition")
        elif partition == "train":
            if selection not in {"train", "validation"}:
                raise ValueError(
                    "train records require a 'train' or 'validation' selection partition"
                )
        else:
            raise ValueError(f"invalid evaluation partition '{partition}'")
        assignment = (partition, selection)
        assignments_seen.add(assignment)

        unique_identities = (
            (hypercube_assignments, seen_hypercubes, row["cropped_hypercube_id"], "CroppedHypercube"),
            (crop_assignments, seen_crops, row["crop_id"], "crop"),
        )
        for assignments, seen, identity, name in unique_identities:
            if not identity:
                raise ValueError(f"missing {name} identity")
            prior = assignments.get(identity)
            if prior is not None and prior != assignment:
                raise ValueError(f"{name} '{identity}' crosses evaluation boundaries")
            if identity in seen:
                raise ValueError(f"duplicate {name} identity '{identity}'")
            assignments[identity] = assignment
            seen.add(identity)
        acquisition_id = row["acquisition_id"]
        if not acquisition_id:
            raise ValueError("missing Acquisition identity")
        prior_assignment = acquisition_assignments.setdefault(acquisition_id, assignment)
        if prior_assignment != assignment:
            raise ValueError(f"Acquisition '{acquisition_id}' crosses evaluation boundaries")
        prior_class = acquisition_classes.setdefault(acquisition_id, row["severity_class"])
        if prior_class != row["severity_class"]:
            raise ValueError(f"conflicting SeverityClass for Acquisition '{acquisition_id}'")
    for assignment, name in (
        (("train", "train"), "train records for band selection"),
        (("train", "validation"), "validation records for band selection"),
        (("test", ""), "held-out test records"),
    ):
        if assignment not in assignments_seen:
            raise ValueError(f"evaluation assignment has no {name}")
    return rows


def load_partitions(path) -> list[dict[str, str]]:
    """Load the evaluation-partition manifest, refusing any leak."""
    rows = read_manifest(path, PARTITION_FIELDS, name="evaluation partition manifest")
    return validate_partition_rows(rows)


# --- Spectral axis ---------------------------------------------------------

SPECTRAL_AXIS_FIELDS = (
    "schema_version",
    "spectral_axis_id",
    "band_count",
    "wavelengths_nm",
)


def json_sha256(value) -> str:
    """Hash a JSON-serializable value under one canonical encoding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def spectral_axis_id(wavelengths) -> str:
    """The content address of an axis: what its `spectral_axis_id` must equal.

    It lets a caller check that the crops it loaded were cut against the same
    axis the wavelengths came from, without reading the manifest again.
    """
    return json_sha256({"wavelengths_nm": tuple(wavelengths)})


def load_spectral_axis(path) -> list[float]:
    """Read the wavelengths, in nm, of the one SpectralAxis the dataset uses.

    The band range of the whole search comes from the length of this list, never
    from a constant.

    Raises:
        ValueError: If the manifest does not hold exactly one axis whose
            wavelengths are as many as its band count, positive, increasing, and
            hashing to the `spectral_axis_id` the manifest claims for them.
    """
    rows = read_manifest(path, SPECTRAL_AXIS_FIELDS, name="SpectralAxis manifest")
    if len(rows) != 1:
        raise ValueError("SpectralAxis manifest must describe exactly one axis")
    row = rows[0]
    try:
        wavelengths = [float(value) for value in json.loads(row["wavelengths_nm"])]
        band_count = int(row["band_count"])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid SpectralAxis definition '{row['spectral_axis_id']}'") from exc
    if (
        not row["spectral_axis_id"]
        or not wavelengths
        or len(wavelengths) != band_count
        or any(not np.isfinite(value) or value <= 0 for value in wavelengths)
        or any(right <= left for left, right in zip(wavelengths, wavelengths[1:], strict=False))
        or spectral_axis_id(wavelengths) != row["spectral_axis_id"]
    ):
        raise ValueError(f"invalid SpectralAxis definition '{row['spectral_axis_id']}'")
    return wavelengths


def wavelengths_of(bands, wavelengths: list[float]) -> tuple[float, float, float]:
    """Translate three distinct in-range band indices to their wavelengths in nm.

    This is the only source of nm: every table, summary and figure that names a
    wavelength gets it from here, in the band order the network sees.

    Raises:
        ValueError: If the values are not exactly three distinct indices inside
            the SpectralAxis.
    """
    try:
        indices = tuple(bands)
    except TypeError as exc:
        raise ValueError("a band triplet requires exactly three distinct in-range indices") from exc
    if (
        len(indices) != 3
        or any(type(index) is not int for index in indices)
        or len(set(indices)) != 3
        or any(index < 0 or index >= len(wavelengths) for index in indices)
    ):
        raise ValueError("a band triplet requires exactly three distinct in-range indices")
    return (wavelengths[indices[0]], wavelengths[indices[1]], wavelengths[indices[2]])


# --- Hypercubes in memory --------------------------------------------------


@dataclass(frozen=True)
class Hypercube:
    """One masked float32 crop, in RAM, with the identity it came from."""

    cropped_hypercube_id: str
    crop_id: str
    acquisition_id: str
    severity_class: int
    spectral_axis_id: str
    reflectance: np.ndarray
    foreground_mask: np.ndarray


def validate_arrays(reflectance: np.ndarray, foreground_mask: np.ndarray) -> None:
    """Refuse arrays that break the CroppedHypercube contract."""
    if (
        reflectance.dtype != np.float32
        or reflectance.ndim != 3
        or any(dimension <= 0 for dimension in reflectance.shape)
        or not np.isfinite(reflectance).all()
        or np.any((reflectance < 0) | (reflectance > 1))
    ):
        raise ValueError(
            "reflectance must be a finite float32 (height, width, bands) array in [0, 1]"
        )
    if foreground_mask.dtype != np.bool_ or foreground_mask.shape != reflectance.shape[:2]:
        raise ValueError("foreground_mask must be boolean and match reflectance spatial shape")
    if not foreground_mask.any():
        raise ValueError("foreground_mask must contain foreground pixels")
    if np.any(reflectance[~foreground_mask] != 0):
        raise ValueError("background reflectance must be zero outside foreground_mask")


def read_npz(path) -> tuple[np.ndarray, np.ndarray]:
    """Read one CroppedHypercube artifact, without unpickling anything."""
    if path.suffix != ".npz":
        raise ValueError("CroppedHypercube artifacts must be NPZ")
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if set(artifact.files) != {"reflectance", "foreground_mask"}:
                raise ValueError("CroppedHypercube NPZ has an incompatible schema")
            reflectance = artifact["reflectance"]
            foreground_mask = artifact["foreground_mask"]
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("CroppedHypercube"):
            raise
        raise ValueError(f"cannot load CroppedHypercube '{path}': {exc}") from exc
    validate_arrays(reflectance, foreground_mask)
    return reflectance, foreground_mask


def artifact_path(manifest_path, entry: dict[str, str]) -> Path:
    """Where one manifest row's NPZ lives, relative to the manifest itself.

    The only place a manifest's `artifact_relative_path` is turned into a path,
    so the checks that keep a hand-edited manifest from reaching outside the
    dataset directory are written once.  `check_data.py` hashes what this
    returns; `load_hypercubes` reads it.

    Raises:
        ValueError: If the recorded path is absolute, climbs out of the
            dataset directory, or is not a POSIX path.
    """
    relative = PurePosixPath(entry["artifact_relative_path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError(
            f"invalid artifact path for CroppedHypercube '{entry['cropped_hypercube_id']}'"
        )
    return Path(manifest_path).parent.joinpath(*relative.parts)


def load_hypercubes(manifest_path, rows: list[dict[str, str]], *, verbose=True) -> list[Hypercube]:
    """Load into RAM the crops named by `rows`, in the order of the manifest.

    The manifest is walked in its own order and filtered by the requested
    identities, never iterated from a set, so two runs load the same arrays in
    the same order.  Loading with train rows only therefore cannot return a test
    crop: what is not asked for is not read.

    Raises:
        ValueError: If an artifact is malformed, if the manifest disagrees with
            the partition rows about a crop's lineage, or if a requested identity
            is not in the manifest.
    """
    manifest_path = Path(manifest_path)
    rows_by_id = {row["cropped_hypercube_id"]: row for row in rows}
    requested = set(rows_by_id)
    manifest = load_manifest(manifest_path)
    hypercubes = []
    for entry in manifest:
        identity = entry["cropped_hypercube_id"]
        if identity not in requested:
            continue
        try:
            expected_shape = (int(entry["height"]), int(entry["width"]), int(entry["band_count"]))
        except ValueError as exc:
            raise ValueError(f"invalid metadata for CroppedHypercube '{identity}'") from exc
        try:
            reflectance, foreground_mask = read_npz(artifact_path(manifest_path, entry))
        except ValueError as exc:
            raise ValueError(f"invalid CroppedHypercube '{identity}': {exc}") from exc
        if reflectance.shape != expected_shape:
            raise ValueError(f"shape mismatch for CroppedHypercube '{identity}'")
        row = rows_by_id[identity]
        if (
            row["crop_id"] != entry["crop_id"]
            or row["acquisition_id"] != entry["acquisition_id"]
            or row["severity_class"] != entry["severity_class"]
        ):
            raise ValueError(f"partition lineage mismatch for CroppedHypercube '{identity}'")
        hypercubes.append(
            Hypercube(
                cropped_hypercube_id=identity,
                crop_id=entry["crop_id"],
                acquisition_id=entry["acquisition_id"],
                severity_class=config.CLASS_NAMES.index(entry["severity_class"]),
                spectral_axis_id=entry["spectral_axis_id"],
                reflectance=reflectance,
                foreground_mask=foreground_mask,
            )
        )
        if verbose and len(hypercubes) % 100 == 0:
            print(f"  loaded {len(hypercubes)}/{len(requested)} hypercubes")
    loaded = {hypercube.cropped_hypercube_id for hypercube in hypercubes}
    if loaded != requested:
        raise ValueError(
            f"CroppedHypercube manifest is missing requested identities: {sorted(requested - loaded)}"
        )
    if verbose and len(hypercubes) % 100 != 0:
        print(f"  loaded {len(hypercubes)}/{len(requested)} hypercubes")
    return hypercubes


def load_partition_crops(
    manifest_path, partition: str, wavelengths: list[float], *, verbose: bool
) -> list[Hypercube]:
    """Load the crops of one evaluation partition, in manifest order.

    `partition` is "train" — all 28 Acquisitions, the inner validation split
    included — or "test".  Asking for one never reads an artifact of the other,
    because the rows of the other are dropped before a single NPZ is opened.

    Raises:
        ValueError: If the crops were cut against another SpectralAxis.
    """
    rows = [row for row in load_partitions(config.PARTITIONS) if row["partition"] == partition]
    if verbose:
        print(f"loading {len(rows)} {partition} crops")
    crops = load_hypercubes(manifest_path, rows, verbose=verbose)
    if {hypercube.spectral_axis_id for hypercube in crops} != {spectral_axis_id(wavelengths)}:
        raise ValueError(f"{partition} crops were cut against another SpectralAxis")
    return crops



# --- Normalization and model input -----------------------------------------


def fit_foreground_normalization(hypercubes: list[Hypercube], indices):
    """Fit per-channel mean and std on training foreground pixels only.

    Background pixels are excluded so masked-out area cannot shift the
    reflectance distribution, and the test crops are simply not here.

    Raises:
        ValueError: If no hypercubes were supplied, or if a channel has no
            variance to divide by.
    """
    if not hypercubes:
        raise ValueError("cannot fit normalization without training hypercubes")
    foreground = np.concatenate(
        [
            hypercube.reflectance[:, :, indices][hypercube.foreground_mask]
            for hypercube in hypercubes
        ]
    )
    mean = foreground.mean(axis=0, dtype=np.float64)
    std = foreground.std(axis=0, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("training foreground normalization has a zero-variance channel")
    return (
        (float(mean[0]), float(mean[1]), float(mean[2])),
        (float(std[0]), float(std[1]), float(std[2])),
    )


def apply_foreground_normalization(hypercube: Hypercube, indices, mean, std):
    """Normalize the selected bands and set the masked background back to zero."""
    image = hypercube.reflectance[:, :, indices].astype(np.float32, copy=True)
    image = (image - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    image[~hypercube.foreground_mask] = 0
    return image, hypercube.foreground_mask.copy()


def prepare_model_input(
    hypercube: Hypercube,
    indices,
    mean,
    std,
    *,
    size,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    rotation_degrees: float = 0,
):
    """Normalize, resize and optionally augment one hypercube for the model.

    The image and its foreground mask are transformed together, so augmentation
    cannot separate a pixel from its mask, and the background is zeroed again
    afterwards so no reflectance is invented where there is no fig.
    """
    image, mask = apply_foreground_normalization(hypercube, indices, mean, std)
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
    mask_tensor = torch.from_numpy(mask[None].astype(np.float32))
    image_tensor = transform_functional.resize(
        image_tensor, list(size), interpolation=InterpolationMode.BILINEAR, antialias=True
    )
    mask_tensor = transform_functional.resize(
        mask_tensor, list(size), interpolation=InterpolationMode.NEAREST
    )
    if horizontal_flip:
        image_tensor = transform_functional.hflip(image_tensor)
        mask_tensor = transform_functional.hflip(mask_tensor)
    if vertical_flip:
        image_tensor = transform_functional.vflip(image_tensor)
        mask_tensor = transform_functional.vflip(mask_tensor)
    if rotation_degrees:
        image_tensor = transform_functional.rotate(
            image_tensor,
            rotation_degrees,
            interpolation=InterpolationMode.BILINEAR,
            fill=[0.0],
        )
        mask_tensor = transform_functional.rotate(
            mask_tensor,
            rotation_degrees,
            interpolation=InterpolationMode.NEAREST,
            fill=[0.0],
        )
    aligned_mask = mask_tensor.squeeze(0) >= 0.5
    image_tensor[:, ~aligned_mask] = 0
    return image_tensor, aligned_mask


class SelectedBandDataset(Dataset):
    """The crops of one partition, as three-channel images of the chosen bands.

    Augmentation decisions are drawn with `torch.rand`, so DataLoader workers
    seeded from the loader's generator reproduce them.
    """

    def __init__(self, hypercubes: list[Hypercube], indices, mean, std, size, *, training: bool):
        self.hypercubes = hypercubes
        self.indices = indices
        self.mean = mean
        self.std = std
        self.size = size
        self.training = training

    def __len__(self) -> int:
        return len(self.hypercubes)

    def __getitem__(self, index: int):
        hypercube = self.hypercubes[index]
        image_tensor, _ = prepare_model_input(
            hypercube,
            self.indices,
            self.mean,
            self.std,
            size=self.size,
            horizontal_flip=self.training and bool(torch.rand(()) < 0.5),
            vertical_flip=self.training and bool(torch.rand(()) < 0.5),
            rotation_degrees=float(torch.empty(()).uniform_(-15, 15)) if self.training else 0,
        )
        return image_tensor, int(hypercube.severity_class)
