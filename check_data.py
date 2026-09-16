"""Verify the linked crops once, and look at what the network is shown.

    uv run python check_data.py
    uv run python check_data.py --bands R G B --manifest another_manifest.csv

Two jobs a one-off utility can do that no step of the chain should pay for on
every load.  The first is hashing every NPZ artifact against the
`artifact_checksum` its CroppedHypercube manifest records, which is how a
symlink pointing at the wrong copy of the dataset is caught: the loaders in
`cnn/data_setup.py` check shape, dtype and lineage on each load and leave the
checksums to this script on purpose.  The second is drawing a few train crops
through exactly the preparation `SelectedBandDataset` hands the network — the
selected bands, the train-foreground normalization, the same resize — so a
mislabelled class or an empty channel is visible rather than inferred.

Nothing here trains, needs a GPU, or writes under `out/`.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib
import numpy as np

import config
from cnn.data_setup import (
    Hypercube,
    artifact_path,
    fit_foreground_normalization,
    load_manifest,
    load_partition_crops,
    load_spectral_axis,
    prepare_model_input,
    wavelengths_of,
)

matplotlib.use("Agg")  # a figure on a headless machine, like every other step
import matplotlib.pyplot as plt  # noqa: E402

READ_BLOCK = 1 << 20  # the NPZ crops are megabytes each; hash them in blocks


# --- The checksums, verified once ------------------------------------------


def artifact_checksum(path) -> str:
    """The SHA-256 of one artifact's bytes, as the manifest records it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for block in iter(lambda: artifact.read(READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def mismatched_artifacts(manifest_path, rows: list[dict[str, str]]) -> list[str]:
    """The identities whose artifact is missing or does not hash as recorded.

    Returned in manifest order, so the report reads the same way twice.

    Raises:
        ValueError: If a row records an artifact path that reaches outside the
            dataset directory.
    """
    mismatched = []
    for row in rows:
        try:
            recorded = artifact_checksum(artifact_path(manifest_path, row))
        except OSError:
            recorded = None
        if recorded != row["artifact_checksum"]:
            mismatched.append(row["cropped_hypercube_id"])
    return mismatched


# --- The grid of crops, as the network sees them ----------------------------


def crops_to_show(hypercubes: list[Hypercube], per_class: int) -> list[Hypercube]:
    """The first `per_class` crops of each class, class by class in label order.

    Raises:
        ValueError: If a class has fewer crops than the grid asks for.
    """
    chosen = []
    for severity, name in enumerate(config.CLASS_NAMES):
        of_class = [
            hypercube for hypercube in hypercubes if hypercube.severity_class == severity
        ][:per_class]
        if len(of_class) < per_class:
            raise ValueError(f"class {name} has fewer than {per_class} train crops to draw")
        chosen.extend(of_class)
    return chosen


BACKGROUND_GREY = 0.72  # outside the fig; not a value any channel can take below


def draw_grid(hypercubes: list[Hypercube], bands, wavelengths: list[float], mean, std, path) -> Path:
    """Save one panel per crop, one class per column, with the nm in the title.

    The panels hold the normalized tensor itself, not the raw reflectance, so
    what is on the page is what the first convolution reads: the three selected
    bands as red, green and blue — false colour, since none of them need be a
    visible wavelength.

    Two choices make the grid readable.  One scale for all the panels, fitted to
    the fig pixels of the whole grid, so the classes stay comparable; a
    per-panel scale would make every crop look alike.  And the masked
    background is painted a flat grey afterwards, outside that scale, because
    `prepare_model_input` sets it to zero in *normalized* space: left alone it
    would land mid-range and a channel that was empty everywhere would be
    indistinguishable from no fig at all.
    """
    prepared = [
        prepare_model_input(hypercube, list(bands), mean, std, size=config.IMAGE_SIZE)
        for hypercube in hypercubes
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

    # One class per column, `per_class` crops down it, which is the order
    # `crops_to_show` returns them in — hence the transpose before flattening.
    columns = config.NUM_CLASSES
    rows = (len(images) + columns - 1) // columns
    # The panels are twice as wide as they are tall, so the row height follows
    # the column width: an aspect-locked image in a taller box is mostly margin.
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.6 * columns, 2.1 * rows), squeeze=False
    )
    panels = axes.T.flatten()
    for axis, hypercube, image in zip(panels, hypercubes, scaled, strict=False):
        axis.imshow(image)
        axis.set_title(
            f"{config.CLASS_NAMES[hypercube.severity_class]}  "
            f"{hypercube.cropped_hypercube_id[:8]}",
            fontsize=10,
        )
        axis.axis("off")
    for axis in panels[len(images) :]:
        axis.axis("off")

    nanometres = " / ".join(f"{value:g} nm" for value in wavelengths_of(tuple(bands), wavelengths))
    figure.suptitle(
        f"Train crops as the network sees them — bands "
        f"{'/'.join(str(band) for band in bands)} = {nanometres} as R/G/B\n"
        "normalized on train foreground, one scale over the fig pixels of the whole grid"
    )
    # Set by hand rather than by `tight_layout`, which packs the rows tightly
    # enough that a panel title reads as a caption of the row above it.
    figure.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02, hspace=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# --- The command ------------------------------------------------------------


def check_command(manifest_path, bands, *, verbose: bool = True) -> int:
    """Verify every artifact, then draw the grid; return the mismatch count.

    The figure is drawn only when nothing mismatched: a grid of crops that are
    not the crops the manifest describes would be a sanity check that lies.

    Raises:
        ValueError: If a manifest is unreadable, or the triplet is not three
            distinct in-range bands.
    """
    manifest_path = Path(manifest_path)
    rows = load_manifest(manifest_path)
    mismatched = mismatched_artifacts(manifest_path, rows)
    noun = "mismatch" if len(mismatched) == 1 else "mismatches"
    print(f"{len(rows)} files, {len(mismatched)} checksum {noun}")
    if mismatched:
        for identity in mismatched:
            print(f"  mismatched: {identity}", file=sys.stderr)
        return len(mismatched)

    wavelengths = load_spectral_axis(config.SPECTRAL_AXES)
    wavelengths_of(tuple(bands), wavelengths)  # refuse a bad triplet before the long load
    crops = load_partition_crops(manifest_path, "train", wavelengths, verbose=verbose)
    shown = crops_to_show(crops, config.SANITY_CHECK_CROPS_PER_CLASS)
    mean, std = fit_foreground_normalization(crops, list(bands))
    figure = draw_grid(
        shown, bands, wavelengths, mean, std, config.SANITY_CHECK_DIR / "data_sanity_check.png"
    )
    print(f"normalization on train foreground: mean {mean}, std {std}")
    print(f"wrote {figure}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=config.HYPERCUBES_MANIFEST,
        help="the CroppedHypercube manifest to verify and draw from",
    )
    parser.add_argument(
        "--bands",
        nargs=3,
        type=int,
        default=list(config.SANITY_CHECK_BANDS),
        metavar=("R", "G", "B"),
        help="the triplet the grid is drawn through; the default is the published winner",
    )
    arguments = parser.parse_args(argv)
    try:
        mismatches = check_command(arguments.manifest, tuple(arguments.bands))
    except ValueError as refusal:
        print(f"check_data.py: {refusal}", file=sys.stderr)
        return 1
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
