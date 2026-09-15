import csv

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
# from tqdm import tqdm
from typing import Optional
from torchvision import transforms
from torch.utils.data import DataLoader

import config

_cpu_count = os.cpu_count()
NUM_WORKERS = _cpu_count if _cpu_count is not None else 0


class HypercubeDataset(Dataset):
    """
    Custom PyTorch Dataset for loading pre-processed hyperspectral data.

    This class is designed to:
    1. Accept pre-loaded hypercube data and labels directly, OR
    2. Read a CSV manifest file containing filepaths and labels and load data.
    3. For a given sample, extract three specified bands to form an RGB-like image.
    4. Apply transformations (e.g., from torchvision) to the resulting image.

    It assumes the .npy files were created by your preprocessing script and
    contain NumPy arrays with a dtype of `uint8` and shape (H, W, C).
    """

    def __init__(self, csv_file=None, band_indices=None, transform=None, 
                 data_samples=None, labels=None, verbose: bool = False):
        """
        Args:
            csv_file (string, optional): Path to the csv file with 'filepath' and 'label' columns.
                                       Used only if data_samples and labels are None.
            band_indices (list or tuple of 3 int): The indices of the three bands
                                                   to use for the R, G, and B channels.
            transform (callable, optional): A function/transform from torchvision
                                            to be applied on a sample.
            data_samples (list, optional): Pre-loaded list of hypercube arrays.
                                         If provided, csv_file is ignored.
            labels (list, optional): Pre-loaded list of labels corresponding to data_samples.
                                   If provided, csv_file is ignored.
            verbose (bool, optional): If True, prints additional information during initialization.
        """
        # --- 1. Store Initialization Arguments ---
        self.band_indices = band_indices
        self.transform = transform
        self.verbose = verbose

        if band_indices is not None and len(band_indices) != 3:
            raise ValueError(
                "`band_indices` must be a list or tuple of exactly 3 integers."
            )

        # --- 2. Use pre-loaded data OR load from CSV ---
        if data_samples is not None and labels is not None:
            # Use pre-loaded data
            if len(data_samples) != len(labels):
                raise ValueError("data_samples and labels must have the same length.")
            
            self.data_samples = data_samples
            self.labels = labels
            if self.verbose:
                print(f"Dataset initialized with pre-loaded data. Total samples: {len(self.labels)}")
            
        elif csv_file is not None:
            # Load from CSV file (original behavior)
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"The specified CSV file was not found: {csv_file}")
            
            annotations = pd.read_csv(csv_file)
            self.data_samples = []
            if self.verbose:
                print(f"Initializing dataset from {csv_file}...")
            for fpath in annotations["filepath"]:  # tqdm removed
                hypercube = np.load(fpath)
                self.data_samples.append(hypercube)
            self.labels = annotations["label"].tolist()
            if self.verbose:
                print(f"Dataset successfully loaded. Total samples: {len(self.labels)}")
            
        else:
            raise ValueError(
                "Either provide csv_file, or both data_samples and labels."
            )

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves one sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed RGB-like tensor
                   and label is the corresponding integer label.
        """
        # --- 1. Retrieve Pre-loaded Data ---
        hypercube = self.data_samples[idx]  # This is a uint8 NumPy array
        label = self.labels[idx]

        # --- 2. Extract Bands to Create an RGB-like Image ---
        # This creates a (Height, Width, 3) NumPy array of dtype uint8.
        if self.band_indices is not None:
            rgb_image = hypercube[:, :, self.band_indices]
        else:
            # If no band_indices specified, assume hypercube already has 3 channels
            if hypercube.shape[2] != 3:
                raise ValueError(
                    f"When band_indices is None, hypercube must have exactly 3 channels, "
                    f"but got {hypercube.shape[2]} channels."
                )
            rgb_image = hypercube

        # --- 3. Apply Transformations ---
        # The `transform` pipeline is crucial. `transforms.ToTensor()` will convert
        # the (H, W, C) uint8 NumPy array into a (C, H, W) float32 Torch tensor
        # and automatically scale the values from the [0, 255] range to [0.0, 1.0].
        if self.transform:
            image_tensor = self.transform(rgb_image)
        else:
            # If no transform is provided, we must manually convert to a tensor and scale.
            image_tensor = (
                torch.from_numpy(rgb_image.transpose((2, 0, 1))).to(torch.float32)
                / 255.0
            )

        return image_tensor, label

    def get_labels(self):
        """Returns the list of labels for all samples in the dataset."""
        return self.labels

    def update_band_indices(self, new_band_indices):
        """
        Update the band indices for RGB extraction.
        
        Args:
            new_band_indices (list or tuple of 3 int): New band indices to use.
        """
        if len(new_band_indices) != 3:
            raise ValueError(
                "`new_band_indices` must be a list or tuple of exactly 3 integers."
            )
        self.band_indices = new_band_indices


def load_hypercubes_from_csv(csv_file, verbose: bool = False):
    """
    Utility function to load all hypercubes from a CSV file into memory.
    
    Args:
        csv_file (str): Path to CSV file with 'filepath' and 'label' columns.
        verbose (bool, optional): If True, prints additional information during loading.
        
    Returns:
        tuple: (data_samples, labels) where data_samples is a list of numpy arrays
               and labels is a list of corresponding labels.
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"The specified CSV file was not found: {csv_file}")
    
    annotations = pd.read_csv(csv_file)
    data_samples = []
    if verbose:
        print(f"Loading hypercubes from {csv_file}...")
    for fpath in annotations["filepath"]:  # tqdm removed
        hypercube = np.load(fpath)
        data_samples.append(hypercube)
    labels = annotations["label"].tolist()
    if verbose:
        print(f"Successfully loaded {len(labels)} hypercubes into memory.")
    return data_samples, labels


def create_train_dataloader(
    train_csv: Optional[str] = None,
    band_indices: Optional[list[int]] = None,
    transform: Optional[transforms.Compose] = None,
    batch_size: int = 32,
    num_workers: int = NUM_WORKERS,
    data_samples: Optional[list] = None,
    labels: Optional[list[int]] = None,
    verbose: bool = False,
):
    """
    Create train dataloader with either CSV file or pre-loaded data.
    
    Args:
        train_csv: Path to CSV file (used if data_samples/labels are None)
        band_indices: List of 3 band indices for RGB channels
        transform: Torchvision transforms
        batch_size: Batch size for dataloader
        num_workers: Number of worker processes
        data_samples: Pre-loaded hypercube data (optional)
        labels: Pre-loaded labels (optional)
        verbose: If True, prints additional information during dataloader creation.
    """
    train_data = HypercubeDataset(
        csv_file=train_csv,
        band_indices=band_indices, 
        transform=transform,
        data_samples=data_samples,
        labels=labels,
        verbose=verbose
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    if verbose:
        print("Band indices for training dataloader:", band_indices)

    return train_dataloader


def create_test_dataloader(
    test_csv: Optional[str] = None,
    band_indices: Optional[list[int]] = None,
    transform: Optional[transforms.Compose] = None,
    batch_size: int = 32,
    num_workers: int = NUM_WORKERS,
    data_samples: Optional[list] = None,
    labels: Optional[list] = None,
    verbose: bool = False,
):
    """
    Create test dataloader with either CSV file or pre-loaded data.
    
    Args:
        test_csv: Path to CSV file (used if data_samples/labels are None)
        band_indices: List of 3 band indices for RGB channels
        transform: Torchvision transforms
        batch_size: Batch size for dataloader
        num_workers: Number of worker processes
        data_samples: Pre-loaded hypercube data (optional)
        labels: Pre-loaded labels (optional)
        verbose: If True, prints additional information during dataloader creation.
    """
    test_data = HypercubeDataset(
        csv_file=test_csv,
        band_indices=band_indices,
        transform=transform,
        data_samples=data_samples,
        labels=labels,
        verbose=verbose
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if verbose:
        print("Band indices for test dataloader:", band_indices)

    return test_dataloader

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


def load_cropped_hypercubes(path) -> list[dict[str, str]]:
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
