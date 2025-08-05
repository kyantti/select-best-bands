"""
Contains functionality for creating PyTorch DataLoaders for
image classification data, including memory-based data loading
for genetic algorithm optimization.
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count() or 1  # Handle None case


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names


# ============================================================================
# MEMORY-BASED DATA LOADING FOR GENETIC ALGORITHM
# ============================================================================


class MemoryDataset(Dataset):
    """Custom Dataset that holds RGB arrays and labels in memory"""

    def __init__(self, rgb_arrays, labels, transform=None):
        """
        Args:
            rgb_arrays: List of RGB numpy arrays (H, W, 3)
            labels: List of corresponding class labels
            transform: Optional transform to be applied on samples
        """
        self.rgb_arrays = rgb_arrays
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.rgb_arrays)

    def __getitem__(self, idx):
        image = self.rgb_arrays[idx]
        label = self.labels[idx]

        # Convert to PIL Image for transforms
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            # Convert to tensor format (C, H, W) and normalize to [0,1]
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        return image, label


def create_rgb_from_patch(patch, r=60, g=30, b=10):
    """Create RGB image from hyperspectral patch using specified bands"""
    rgb = patch[:, :, [r, g, b]]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


def load_hypercubes_to_memory(input_base_dir="data/interim/cropped-hypercubes"):
    """
    Load all hypercube patches into memory for fast RGB generation during GA.

    Returns:
        dict: {class_name: [list of hypercube patches]}
    """
    print("🔄 Loading all hypercube patches into memory...")

    if not os.path.exists(input_base_dir):
        raise FileNotFoundError(f"Input directory {input_base_dir} does not exist!")

    hypercubes = {}
    total_patches = 0

    # Get all class subdirectories (C0, C1, C2, C3, etc.)
    subdirs = [
        d
        for d in os.listdir(input_base_dir)
        if os.path.isdir(os.path.join(input_base_dir, d))
    ]
    subdirs.sort()

    for subdir in subdirs:
        class_dir = os.path.join(input_base_dir, subdir)
        patch_files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]

        print(f"📂 Loading {len(patch_files)} patches from class {subdir}...")

        class_patches = []
        for file in patch_files:
            patch_path = os.path.join(class_dir, file)
            patch = np.load(patch_path)
            class_patches.append(patch)
            total_patches += 1

        hypercubes[subdir] = class_patches

    print(f"✅ Loaded {total_patches} hypercube patches into memory")
    print(f"📊 Classes found: {list(hypercubes.keys())}")

    return hypercubes


def generate_rgb_arrays_from_bands(hypercubes, r_band, g_band, b_band):
    """
    Generate RGB arrays from hypercubes using specified bands.

    Args:
        hypercubes: Dict from load_hypercubes_to_memory()
        r_band, g_band, b_band: Band indices for RGB channels

    Returns:
        tuple: (rgb_arrays, labels, class_names)
    """
    rgb_arrays = []
    labels = []
    class_names = sorted(hypercubes.keys())

    print(f"🎨 Generating RGB arrays using bands R:{r_band}, G:{g_band}, B:{b_band}")

    for class_idx, class_name in enumerate(class_names):
        patches = hypercubes[class_name]

        for patch in patches:
            rgb = create_rgb_from_patch(patch, r_band, g_band, b_band)
            rgb_arrays.append(rgb)
            labels.append(class_idx)

    print(f"✅ Generated {len(rgb_arrays)} RGB arrays")
    return rgb_arrays, labels, class_names


def create_memory_dataloaders(
    hypercubes,
    r_band,
    g_band,
    b_band,
    train_ratio=0.8,
    batch_size=32,
    transform=None,
    random_seed=42,
    image_size=(224, 224),
):
    """
    Create train/test dataloaders from hypercubes in memory.

    Args:
        hypercubes: Dict from load_hypercubes_to_memory()
        r_band, g_band, b_band: Band indices for RGB channels
        train_ratio: Proportion of data for training
        batch_size: Batch size for dataloaders
        transform: Optional transform for data augmentation
        random_seed: Random seed for reproducible splits
        image_size: Target size for all images (height, width)

    Returns:
        tuple: (train_dataloader, test_dataloader, class_names)
    """
    # Generate RGB arrays
    rgb_arrays, labels, class_names = generate_rgb_arrays_from_bands(
        hypercubes, r_band, g_band, b_band
    )

    # Create default transform if none provided (resize to ensure consistent size)
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),  # Resize all images to same size
                transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
            ]
        )

    # Create train/test split
    np.random.seed(random_seed)
    indices = np.random.permutation(len(rgb_arrays))
    split_idx = int(len(rgb_arrays) * train_ratio)

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Split data
    train_rgb = [rgb_arrays[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_rgb = [rgb_arrays[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    # Create datasets
    train_dataset = MemoryDataset(train_rgb, train_labels, transform)
    test_dataset = MemoryDataset(test_rgb, test_labels, transform)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
    )

    print(f"📊 Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    return train_dataloader, test_dataloader, class_names
