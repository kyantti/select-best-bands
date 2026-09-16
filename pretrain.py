"""SimCLR pretraining of the backbone on band-triplet views of the train crops.

Rewritten in this repository's shape from fig-aflatoxin's
`scripts/contrastive-pretrain.py`, the one intervention of that record that beat
its own prospective criterion (+0.023 pooled, positive on 8 of 8 candidates).
Two views of a crop are two *different* random band triplets of that same crop,
put through `prepare_model_input`, the preparation the network sees at
evaluation time.  The pretext task is therefore the downstream family: represent
a fig well whatever three bands it is shown.  There are no labels and no
classification head, only the backbone and a projection head thrown away with
the loss.

**Two checkpoints, never one.**  `--cohort fit` sees the 22 train-fit
Acquisitions and is the backbone a *search* may start from; `--cohort train`
sees all 28 train Acquisitions and is the one the *final model* starts from.
A search initialized from a backbone that had seen the validation crops would
optimize a fitness contaminated by the very set it is scored on, and neither
cohort may ever contain a test Acquisition: only train rows are loaded, the same
way `ga.py` loads them, so what is not in the cohort is never opened.

Each checkpoint is a flat backbone state dict that `cnn/model.py`'s injection
point loads as it is, beside a JSON sidecar naming the Acquisitions it saw.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python pretrain.py --cohort fit
    CUDA_VISIBLE_DEVICES=1 uv run python pretrain.py --cohort train
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

import config
import ga
from cnn.data_setup import (
    Hypercube,
    load_hypercubes,
    load_partitions,
    load_spectral_axis,
    prepare_model_input,
    spectral_axis_id,
)
from cnn.model import (
    HEAD_PREFIX,
    RESNET50_WEIGHTS,
    build_resnet50,
    checkpoint_identity,
    hypercubes_checksum,
    seed_everything,
)

Triplet = tuple[int, int, int]


# --- The two cohorts --------------------------------------------------------


def cohort_rows(rows: list[dict[str, str]], cohort: str) -> list[dict[str, str]]:
    """The partition rows one cohort pretrains on: train rows, and nothing else.

    `fit` keeps the Acquisitions the search fits on, so a backbone built from it
    has never seen the crops its fitness is scored on; `train` keeps all 28 train
    Acquisitions, which is what the final model is allowed to start from.  A test
    row is in neither, and it is dropped here — before a single artifact is
    opened — rather than skipped later.

    Raises:
        ValueError: If the cohort is not one this protocol has.
    """
    if cohort not in config.PRETRAIN_COHORTS:
        raise ValueError(
            f"'{cohort}' is not a pretraining cohort; "
            f"it is one of {', '.join(config.PRETRAIN_COHORTS)}"
        )
    train = [row for row in rows if row["partition"] == "train"]
    if cohort == "fit":
        return [row for row in train if row["selection_partition"] == "train"]
    return train


def load_cohort(cohort: str, *, verbose: bool = True) -> tuple[list[Hypercube], list[float]]:
    """Load one cohort's crops, in manifest order, with the SpectralAxis they were cut against.

    Mirrors `ga.load_selection_data`: the rows that do not belong are dropped
    first, so the crops of another partition are not read at all.

    Raises:
        ValueError: If the manifests leak across a boundary, if an artifact is
            malformed, or if the crops were cut against another SpectralAxis.
    """
    rows = cohort_rows(load_partitions(config.PARTITIONS), cohort)
    if verbose:
        acquisitions = len({row["acquisition_id"] for row in rows})
        print(f"loading {len(rows)} {cohort} crops from {acquisitions} acquisitions")
    crops = load_hypercubes(config.HYPERCUBES_MANIFEST, rows, verbose=verbose)
    wavelengths = load_spectral_axis(config.SPECTRAL_AXES)
    if {hypercube.spectral_axis_id for hypercube in crops} != {spectral_axis_id(wavelengths)}:
        raise ValueError(f"{cohort} crops were cut against another SpectralAxis")
    return crops, wavelengths


# --- Normalization over every band ------------------------------------------


def per_band_foreground_statistics(
    hypercubes: list[Hypercube],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of every band's foreground pixels, over the whole cohort.

    The evaluator fits its three channels with `fit_foreground_normalization`;
    here the triplet changes with every view, so the cohort is accumulated once
    over all its bands and a triplet's statistics are sliced out of it.  The two
    agree to well inside the 1e-6 the protocol test holds them to.

    Raises:
        ValueError: If there are no hypercubes, or if a band has no foreground
            variance to divide by.
    """
    if not hypercubes:
        raise ValueError("cannot fit per-band normalization without hypercubes")
    bands = hypercubes[0].reflectance.shape[2]
    total = np.zeros(bands, dtype=np.float64)
    total_squared = np.zeros(bands, dtype=np.float64)
    count = 0
    for hypercube in hypercubes:
        pixels = hypercube.reflectance[hypercube.foreground_mask].astype(np.float64)
        total += pixels.sum(axis=0)
        total_squared += (pixels**2).sum(axis=0)
        count += pixels.shape[0]
    mean = total / count
    variance = total_squared / count - mean**2
    std = np.sqrt(np.clip(variance, 0, None))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("cohort foreground normalization has a zero-variance band")
    return mean, std


def triplet_statistics(mean: np.ndarray, std: np.ndarray, bands: Triplet):
    """The three-channel mean and std of one triplet, sliced out of the cohort's."""
    return (
        tuple(float(mean[band]) for band in bands),
        tuple(float(std[band]) for band in bands),
    )


# --- The two views of a crop ------------------------------------------------


def draw_band_triplet(band_count: int) -> Triplet:
    """Three distinct in-range bands, drawn from the torch generator.

    A permutation, not three independent draws: a view whose channels repeat a
    band is a two-channel image wearing three, which is the same guarantee the
    genetic operators repair into every candidate.
    """
    return tuple(int(band) for band in torch.randperm(band_count)[:3].tolist())


def draw_two_views(band_count: int) -> tuple[Triplet, Triplet]:
    """The two band triplets one crop is shown as, never the same triplet twice.

    Two independent draws land on the same triplet about once in ninety million,
    and that pair would ask the loss to pull an image towards itself, teaching
    nothing.  Redrawing costs nothing and makes "two views" true rather than
    almost always true.

    Raises:
        ValueError: If the axis has fewer than four bands, which is the only
            case where "two different triplets" can be impossible to draw.
    """
    if band_count < 4:
        raise ValueError("two different band triplets need an axis of at least four bands")
    left = draw_band_triplet(band_count)
    right = draw_band_triplet(band_count)
    while right == left:
        right = draw_band_triplet(band_count)
    return left, right


class BandTripletViewDataset(Dataset):
    """Each crop as two views: two random band triplets of that same crop.

    The preparation is the evaluator's own — normalize, resize, flips, rotation,
    image and mask moving together and the background zeroed again afterwards —
    so what the backbone learns to represent is what it is later fine-tuned on.
    The bands come from `torch.randperm` and the augmentation from `torch.rand`,
    as `SelectedBandDataset`'s does, so workers seeded from the loader's
    generator reproduce every draw.
    """

    def __init__(self, hypercubes: list[Hypercube], mean: np.ndarray, std: np.ndarray, size):
        self.hypercubes = hypercubes
        self.mean = mean
        self.std = std
        self.size = size
        self.band_count = len(mean)

    def __len__(self) -> int:
        return len(self.hypercubes)

    def view(self, hypercube: Hypercube, bands: Triplet) -> torch.Tensor:
        """One augmented three-channel view of `hypercube` through `bands`."""
        mean, std = triplet_statistics(self.mean, self.std, bands)
        image, _ = prepare_model_input(
            hypercube,
            bands,
            mean,
            std,
            size=self.size,
            horizontal_flip=bool(torch.rand(()) < 0.5),
            vertical_flip=bool(torch.rand(()) < 0.5),
            rotation_degrees=float(torch.empty(()).uniform_(-15, 15)),
        )
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        hypercube = self.hypercubes[index]
        left, right = draw_two_views(self.band_count)
        return self.view(hypercube, left), self.view(hypercube, right)


# --- The objective and the model --------------------------------------------


def nt_xent(projected: torch.Tensor, temperature: float) -> torch.Tensor:
    """SimCLR's normalized temperature-scaled cross entropy over 2N views.

    `projected` is the two halves of one batch stacked, view `i` of the first
    half pairing with view `i + N` of the second.  Every other view in the batch
    is a negative, including the other views of other crops.
    """
    batch = projected.shape[0] // 2
    features = torch.nn.functional.normalize(projected, dim=1)
    similarity = features @ features.T / temperature
    # A view is its own nearest neighbour and would win every time: it is removed
    # from the denominator rather than merely not rewarded.
    similarity.fill_diagonal_(float("-inf"))
    targets = torch.arange(2 * batch, device=projected.device)
    targets = torch.where(targets < batch, targets + batch, targets - batch)
    return torch.nn.functional.cross_entropy(similarity, targets)


def build_pretraining_model(device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    """The evaluator's own backbone, fully trainable, under a projection head.

    `build_resnet50` is called rather than copied, so the object adapted here is
    the object the fine-tuning will load the result back into.  Its four-class
    head is dropped: there are no labels to predict, and the downstream head is
    drawn fresh whatever this run does.  Where `build_resnet50` leaves only
    `layer4` trainable, everything here is unfrozen: a fine-tuning run loads this
    checkpoint into every layer, the frozen ones included, so every layer is
    worth adapting.
    """
    model = build_resnet50(device)
    for parameter in model.parameters():
        parameter.requires_grad = True
    features = int(model.fc.in_features)
    model.fc = torch.nn.Identity()
    projector = torch.nn.Sequential(
        torch.nn.Linear(features, config.PRETRAIN_PROJECTION_HIDDEN),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(config.PRETRAIN_PROJECTION_HIDDEN, config.PRETRAIN_PROJECTION_DIM),
    ).to(device)
    return model, projector


# --- What the run is handed, and what it hands back -------------------------


@dataclass(frozen=True)
class Pretraining:
    """Everything the pretraining run sees, and every setting it runs under.

    There is no test crop in it and no field that could lead to one: the rows of
    the other partitions were dropped before any artifact was opened.

    The hyperparameters live here rather than in the training function's
    defaults so that the sidecar can record what this run was actually given.
    A record that quoted `config` instead would describe the run the constants
    happen to name today, which is not the same thing as the run that produced
    the weights beside it.
    """

    cohort: str
    crops: list[Hypercube]
    mean: np.ndarray
    std: np.ndarray
    seed: int
    device: torch.device
    epochs: int
    batch_size: int = config.PRETRAIN_BATCH_SIZE
    learning_rate: float = config.PRETRAIN_LEARNING_RATE
    temperature: float = config.PRETRAIN_TEMPERATURE
    size: tuple[int, int] = config.IMAGE_SIZE
    num_workers: int = config.NUM_WORKERS


@dataclass(frozen=True)
class PretrainedBackbone:
    """The adapted backbone, on the CPU, and one NT-Xent row per epoch."""

    state: dict[str, torch.Tensor]
    history: list[dict]


def pretrain_backbone(context: Pretraining, *, verbose: bool = True) -> PretrainedBackbone:
    """Adapt the ImageNet backbone contrastively on the cohort's own crops.

    The generators are seeded first, the loader is built next and the model after
    it, the order every training run in this repository is built in, so the same
    seed and the same cohort give back the same weights.

    Raises:
        ValueError: If the cohort holds fewer crops than one batch, so that
            `drop_last` would leave the loop nothing to train on.
    """
    generator = seed_everything(context.seed)
    loader = DataLoader(
        BandTripletViewDataset(context.crops, context.mean, context.std, context.size),
        batch_size=context.batch_size,
        shuffle=True,
        num_workers=context.num_workers,
        persistent_workers=context.num_workers > 0,
        # The loss reads every other view in the batch as a negative, so a short
        # last batch would score its epoch under a different objective.
        drop_last=True,
        pin_memory=context.device.type == "cuda",
        generator=generator,
    )
    if len(loader) == 0:
        raise ValueError(
            f"the {context.cohort} cohort has {len(context.crops)} crops, "
            f"fewer than the batch size of {context.batch_size}"
        )
    model, projector = build_pretraining_model(context.device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()), lr=context.learning_rate
    )
    scaler = GradScaler(device="cuda") if context.device.type == "cuda" else None
    model.train()
    projector.train()
    history = []
    for epoch in range(1, context.epochs + 1):
        total, batches = 0.0, 0
        for left, right in loader:
            views = torch.cat([left, right]).to(context.device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=context.device.type == "cuda",
            ):
                projected = projector(model(views))
            # Outside the autocast block on purpose, which is the one place this
            # rewrite departs from fig-aflatoxin's script: there the loss was
            # computed inside it, so the similarity matrix ran in float16
            # whatever the `.float()` said.  NT-Xent exponentiates a similarity
            # divided by 0.2, and float16 saturates well before that is finished
            # with.  Ticket 15's paired gate measures this loop on its own
            # numbers, so it is the correct version that gets measured.
            loss = nt_xent(projected.float(), context.temperature)
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total += loss.item()
            batches += 1
        history.append({"epoch": epoch, "nt_xent": total / batches})
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == context.epochs):
            print(
                f"  epoch {epoch:>4}/{context.epochs}  nt_xent {total / batches:.4f}",
                flush=True,
            )
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return PretrainedBackbone(state, history)


# --- The checkpoint and its sidecar -----------------------------------------


def pretrain_paths(cohort: str, output: Path | None = None) -> dict[str, Path]:
    """Where one cohort's checkpoint and its sidecar go.

    The sidecar sits beside the weights under the same stem, so a checkpoint
    moved or copied without its record is visibly missing it.

    Raises:
        ValueError: If an output path was given that is not a `.pt` file.
    """
    path = Path(output) if output is not None else config.MODELS_DIR / f"pretrain_{cohort}.pt"
    if path.suffix != ".pt":
        raise ValueError(f"a backbone checkpoint is a .pt file, not '{path.name}'")
    return {"checkpoint": path, "sidecar": path.with_suffix(".json")}


def sidecar(
    context: Pretraining, *, history: list[dict], wavelengths: list[float], identity: dict
) -> dict:
    """What the checkpoint saw, and under what settings.

    The Acquisition identities are the leakage record: a reader checks them
    against the partition manifest and sees, without loading one crop, that no
    test Acquisition — and for the search's backbone, no validation Acquisition
    — went into these weights.
    """
    acquisitions = sorted({crop.acquisition_id for crop in context.crops})
    return {
        "cohort": context.cohort,
        "acquisition_ids": acquisitions,
        "acquisition_count": len(acquisitions),
        "crop_count": len(context.crops),
        "cropped_hypercubes": hypercubes_checksum(context.crops),
        "seed": context.seed,
        "epochs": context.epochs,
        "batch_size": context.batch_size,
        "learning_rate": context.learning_rate,
        "temperature": context.temperature,
        "projection_hidden": config.PRETRAIN_PROJECTION_HIDDEN,
        "projection_dim": config.PRETRAIN_PROJECTION_DIM,
        "view_policy": config.PRETRAIN_VIEW_POLICY,
        "image_size": list(context.size),
        # Recorded because the augmentation stream is split across the workers:
        # the same seed at another worker count is another set of views, and so
        # another backbone.
        "num_workers": context.num_workers,
        "band_count": len(wavelengths),
        "device_type": context.device.type,
        "partition_seed": config.PARTITION_SEED,
        "validation_balance": config.VALIDATION_BALANCE,
        "resnet50_weights": str(RESNET50_WEIGHTS),
        "libraries": ga.library_versions(),
        "nt_xent": [row["nt_xent"] for row in history],
        "checkpoint": identity,
    }


def write_sidecar(path: Path, record: dict) -> None:
    """Write the record beside the weights, sorted and newline-terminated."""
    with path.open("w") as target:
        json.dump(record, target, indent=2, sort_keys=True)
        target.write("\n")


# --- One cohort, end to end -------------------------------------------------


def pretrain_command(
    cohort: str,
    *,
    epochs: int = config.PRETRAIN_EPOCHS,
    seed: int = config.PRETRAIN_SEED,
    output: Path | None = None,
    force: bool = False,
    trainer=pretrain_backbone,
    verbose: bool = True,
) -> None:
    """Pretrain one cohort's backbone and write it with its sidecar.

    Raises:
        ValueError: If the cohort is not one this protocol has, if the device is
            not there, if the crops leak or are malformed, or if the trainer
            hands back something that is not a backbone with one row per epoch.
        FileExistsError: If the checkpoint is already there and `force` is not
            set.  Checked before anything is loaded, so three minutes of GPU are
            never spent on a file that cannot be written.
    """
    paths = pretrain_paths(cohort, output)
    if paths["checkpoint"].exists() and not force:
        raise FileExistsError(
            f"{paths['checkpoint']} is already there. A checkpoint is part of the fitness "
            "contract of every candidate trained from it, so rewriting one makes those "
            "numbers unreproducible: pass --force if that is what you mean, or --output "
            "another path."
        )
    device = ga.select_device()
    crops, wavelengths = load_cohort(cohort, verbose=verbose)
    mean, std = per_band_foreground_statistics(crops)
    context = Pretraining(cohort, crops, mean, std, seed, device, epochs)
    if verbose:
        acquisitions = sorted({crop.acquisition_id for crop in crops})
        print(f"cohort         {cohort}")
        print(f"crops          {len(crops)} in {len(acquisitions)} acquisitions")
        print(f"bands          {len(wavelengths)}")
        print(f"device         {device}  |  epochs {epochs}  |  seed {seed}")
        print(f"view policy    {config.PRETRAIN_VIEW_POLICY}")

    started = time.perf_counter()
    backbone = trainer(context)
    if not backbone.state or len(backbone.history) != epochs:
        raise ValueError("the pretrainer returned no weights, or not one history row per epoch")
    # The downstream head belongs to this experiment's four classes and is drawn
    # fresh every time; a checkpoint that carried one would be offering weights
    # nothing is allowed to load.
    if any(name.startswith(HEAD_PREFIX) for name in backbone.state):
        raise ValueError("a pretraining checkpoint never carries a classification head")
    elapsed = time.perf_counter() - started

    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone.state, paths["checkpoint"])
    identity = checkpoint_identity(paths["checkpoint"])
    write_sidecar(
        paths["sidecar"],
        sidecar(context, history=backbone.history, wavelengths=wavelengths, identity=identity),
    )
    if verbose:
        first, last = backbone.history[0]["nt_xent"], backbone.history[-1]["nt_xent"]
        print(f"nt_xent        {first:.4f} -> {last:.4f}  ({elapsed:.1f} s)")
        print(f"checkpoint     {paths['checkpoint']}")
        print(f"checkpoint sha {identity['sha256']}")
        print(f"sidecar        {paths['sidecar']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cohort",
        required=True,
        choices=config.PRETRAIN_COHORTS,
        help="'fit' for the backbone a search may start from (the 22 train-fit"
        " acquisitions), 'train' for the one the final model starts from (all 28)",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.PRETRAIN_EPOCHS, help="epochs to adapt for"
    )
    parser.add_argument(
        "--seed", type=int, default=config.PRETRAIN_SEED, help="the seed the run is drawn from"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="where to write the .pt; the default is out/models/pretrain_<cohort>.pt"
        " and the sidecar always goes beside it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rewrite a checkpoint that is already there, invalidating every"
        " cached fitness that was trained from it",
    )
    arguments = parser.parse_args(argv)

    # A refused device, an occupied path or a leaking manifest is a decision this
    # program made on purpose, so it reads as one line rather than a traceback.
    try:
        pretrain_command(
            arguments.cohort,
            epochs=arguments.epochs,
            seed=arguments.seed,
            output=arguments.output,
            force=arguments.force,
        )
    except (ValueError, LookupError, FileExistsError) as refusal:
        print(f"pretrain.py: {refusal}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
