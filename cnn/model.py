"""The ResNet50 contract: data identity, per-candidate seed, training, metrics.

Ported from fig-aflatoxin's evaluation module and its evaluate stage.  What is
copied literally here is everything a fitness depends on: the order the four
random number generators are seeded in, how the two DataLoaders are built, the
fact that the model is created *after* them, and the hash that turns a
candidate plus the data it trains on into a training seed.  Change any of it
and the numbers of the 10 Sep 2026 run stop reproducing.
"""

import hashlib
import random

import numpy as np
import torch
import torchvision
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    recall_score,
)
from torch.utils.data import DataLoader

import config
from cnn import engine
from cnn.data_setup import (
    Hypercube,
    SelectedBandDataset,
    fit_foreground_normalization,
    json_sha256,
    spectral_axis_id,
)

# The candidate seed is a contract, not an implementation detail: it names the
# rule a triplet's seed was derived under, so a seed recorded beside a fitness
# can be told apart from one derived under another rule.
SEED_POLICY = "sha256-candidate-split-v1"

RESNET50_WEIGHTS = torchvision.models.ResNet50_Weights.DEFAULT


# --- Data identity ---------------------------------------------------------


def array_identity(array: np.ndarray) -> dict:
    """The content address of one array: what it holds, not where it came from."""
    return {
        "dtype": str(array.dtype),
        "shape": array.shape,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def hypercubes_checksum(hypercubes: list[Hypercube]) -> str:
    """Hash a set of crops by content, independent of the order they were loaded in.

    fig-aflatoxin calls this `_hypercube_identities`; the dict it hashes is the
    same one, so the two can still be read side by side.
    """
    return json_sha256(
        [
            {
                "acquisition_id": item.acquisition_id,
                "crop_id": item.crop_id,
                "cropped_hypercube_id": item.cropped_hypercube_id,
                "foreground_mask": array_identity(item.foreground_mask),
                "reflectance": array_identity(item.reflectance),
                "severity_class": config.CLASS_NAMES[item.severity_class],
                "spectral_axis_id": item.spectral_axis_id,
            }
            for item in sorted(hypercubes, key=lambda value: value.cropped_hypercube_id)
        ]
    )


def data_identity(
    training: list[Hypercube],
    validation: list[Hypercube],
    train_rows: list[dict[str, str]],
    wavelengths: list[float],
) -> dict[str, str]:
    """Checksum the data a candidate is scored on: crops, assignments and axis.

    Held-out test crops are not in it because they are never loaded.  The two
    cohorts are hashed apart, so moving a crop from train to validation is a
    different identity even though the union has not changed.
    """
    return {
        "train_cropped_hypercubes": hypercubes_checksum(training),
        "validation_cropped_hypercubes": hypercubes_checksum(validation),
        "train_evaluation_assignments": json_sha256(
            sorted(train_rows, key=lambda row: row["cropped_hypercube_id"])
        ),
        "spectral_axis": json_sha256(
            {
                "spectral_axis_id": spectral_axis_id(wavelengths),
                "wavelengths_nm": list(wavelengths),
            }
        ),
    }


def candidate_seed(
    bands: tuple[int, int, int], identity: dict[str, str], seed: int = config.CANDIDATE_SEED
) -> int:
    """Derive the training seed of one candidate from the candidate and its data.

    A triplet therefore trains identically whichever generation it turns up in,
    and the same triplet against different data is a different run.
    """
    fingerprint = json_sha256(
        {
            "candidate": [int(band) for band in bands],
            "input_checksums": identity,
            "seed": seed,
            "seed_policy": SEED_POLICY,
        }
    )
    return int(fingerprint[:8], 16)


# --- Metrics ---------------------------------------------------------------


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Report weighted and macro F1, ordinal MAE, quadratic kappa and recall.

    Weighted F1 is the primary metric: each class's F1 counts in proportion to
    its support, so the score follows the class mix of the partition it is
    measured on.  Macro F1 is kept beside it because every earlier experiment
    reports it.  The labels are always the four classes, named explicitly, so a
    class that no crop was predicted into scores zero instead of disappearing.
    """
    labels = list(range(config.NUM_CLASSES))
    return {
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "ordinal_mae": float(mean_absolute_error(y_true, y_pred)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
        ),
        "per_class_recall": {
            name: float(value)
            for name, value in zip(
                config.CLASS_NAMES,
                recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
                strict=True,
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# --- The model -------------------------------------------------------------


def build_resnet50(device: torch.device) -> torch.nn.Module:
    """ResNet50 with pretrained weights, `layer4` and a fresh 4-class head unfrozen."""
    model = torchvision.models.resnet50(weights=RESNET50_WEIGHTS)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = torch.nn.Linear(model.fc.in_features, config.NUM_CLASSES, bias=True)
    return model.to(device)


def seed_everything(seed: int) -> torch.Generator:
    """Seed the four generators a training run draws from, and return the loader's.

    The order matters: `random`, NumPy, torch and CUDA are seeded before any
    DataLoader or module is created, because both consume randomness as they
    are built.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return torch.Generator().manual_seed(seed)


def evaluate_candidate(
    bands: tuple[int, int, int],
    training: list[Hypercube],
    validation: list[Hypercube],
    *,
    seed: int,
    device: torch.device,
    epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    size: tuple[int, int] = config.IMAGE_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> tuple[list[int], dict, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Fit one band triplet on `training` and score it on `validation`.

    Normalization is fitted on the training crops alone, so the crops that
    grade the candidate contribute nothing to the scale it is graded under.
    Returns the ordered validation predictions, the metrics, and the frozen
    mean and std.

    Raises:
        ValueError: If the training crops have a channel with no variance to
            divide by, propagated from `fit_foreground_normalization`.
    """
    indices = tuple(int(band) for band in bands)
    mean, std = fit_foreground_normalization(training, indices)
    generator = seed_everything(seed)
    train_loader = DataLoader(
        SelectedBandDataset(training, indices, mean, std, size, training=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    # No generator here on purpose: the validation loader never shuffles and
    # never augments, so giving it one would only move the training stream.
    validation_loader = DataLoader(
        SelectedBandDataset(validation, indices, mean, std, size, training=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    # The model is built after the loaders: it draws from the same torch
    # generator when its head is initialized.
    model = build_resnet50(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    engine.train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=validation_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        verbose=False,
        device=device,
    )
    predictions = predict(model, validation_loader, device)
    truth = [int(item.severity_class) for item in validation]
    return predictions, classification_metrics(truth, predictions), (mean, std)


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> list[int]:
    """One ordered inference pass: prediction `i` belongs to the loader's crop `i`."""
    predictions: list[int] = []
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:
            predictions.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
    return predictions
