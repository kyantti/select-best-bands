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
from torchvision import transforms
from sklearn.model_selection import train_test_split
import shutil

NUM_WORKERS = os.cpu_count() or 1  # Handle None case

# Global variables to store hypercubes in memory
TRAIN_HYPERCUBES = None
TEST_HYPERCUBES = None
CLASS_NAMES = None


def load_hypercubes_to_memory(train_dir: str, test_dir: str):
    """
    Load all train and test hypercubes into memory for fast band selection.
    
    Args:
        train_dir: Path to training directory containing class subdirectories with .npy files
        test_dir: Path to testing directory containing class subdirectories with .npy files
        
    Returns:
        tuple: (train_hypercubes, test_hypercubes, class_names)
            train_hypercubes: List of (hypercube, class_idx) tuples
            test_hypercubes: List of (hypercube, class_idx) tuples
            class_names: Sorted list of class names
    """
    global TRAIN_HYPERCUBES, TEST_HYPERCUBES, CLASS_NAMES
    
    print("🔄 Loading all hypercubes into memory...")
    
    # Get class names from train directory
    class_names = [d for d in os.listdir(train_dir) 
                   if os.path.isdir(os.path.join(train_dir, d))]
    class_names.sort()
    
    print(f"📊 Classes found: {class_names}")
    
    # Load training hypercubes
    train_hypercubes = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(train_dir, class_name)
        npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        print(f"📂 Loading {len(npy_files)} training files from class {class_name}")
        
        for npy_file in npy_files:
            file_path = os.path.join(class_dir, npy_file)
            hypercube = np.load(file_path)
            train_hypercubes.append((hypercube, class_idx))
    
    # Load testing hypercubes
    test_hypercubes = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        print(f"📂 Loading {len(npy_files)} testing files from class {class_name}")
        
        for npy_file in npy_files:
            file_path = os.path.join(class_dir, npy_file)
            hypercube = np.load(file_path)
            test_hypercubes.append((hypercube, class_idx))
    
    print(f"✅ Loaded {len(train_hypercubes)} training and {len(test_hypercubes)} testing hypercubes into memory")
    
    # Store in global variables for reuse
    TRAIN_HYPERCUBES = train_hypercubes
    TEST_HYPERCUBES = test_hypercubes
    CLASS_NAMES = class_names
    
    return train_hypercubes, test_hypercubes, class_names


def generate_rgb_from_memory(hypercubes, r_band: int, g_band: int, b_band: int):
    """
    Generate RGB arrays from in-memory hypercubes using specified bands.
    
    Args:
        hypercubes: List of (hypercube, class_idx) tuples from load_hypercubes_to_memory()
        r_band: Red channel band index
        g_band: Green channel band index
        b_band: Blue channel band index
        
    Returns:
        tuple: (rgb_arrays, labels)
            rgb_arrays: List of RGB numpy arrays (H, W, 3)
            labels: List of corresponding class indices
    """
    rgb_arrays = []
    labels = []
    
    for hypercube, class_idx in hypercubes:
        rgb_image = create_rgb_from_patch(hypercube, r_band, g_band, b_band)
        
        # Ensure uint8 format
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
        
        rgb_arrays.append(rgb_image)
        labels.append(class_idx)
    
    return rgb_arrays, labels


def split_dataset(source_dir, train_dir, test_dir, test_ratio=0.2):
    """Splits data from source_dir into train_dir and test_dir based on test_ratio.

    Args:
        source_dir (str): Path to the source directory containing class subdirectories.
        train_dir (str): Path to the directory where training data will be stored.
        test_dir (str): Path to the directory where testing data will be stored.
        test_ratio (float): Proportion of data to allocate to the test set (default: 0.2).
    """
    # Ensure train_dir and test_dir exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate over each class subdirectory in source_dir
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Create corresponding class subdirectories in train_dir and test_dir
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Get all file paths in the class directory
        file_paths = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Split file paths into train and test sets
        train_files, test_files = train_test_split(file_paths, test_size=test_ratio, random_state=42)

        # Move files to their respective directories
        for file_path in train_files:
            shutil.copy(file_path, train_class_dir)
        for file_path in test_files:
            shutil.copy(file_path, test_class_dir)


def create_dataloaders(
    r_band: int,
    g_band: int,
    b_band: int,
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    use_memory: bool = True,
):
    """Creates training and testing DataLoaders from numpy arrays.

    Takes in training and testing directory paths containing numpy arrays
    organized by class and creates PyTorch DataLoaders.

    Args:
        r_band: Red channel band index.
        g_band: Green channel band index.
        b_band: Blue channel band index.
        train_dir: Path to training directory containing class subdirectories with .npy files.
        test_dir: Path to testing directory containing class subdirectories with .npy files.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.
        use_memory: If True, use in-memory hypercubes for faster band extraction.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = \
            create_dataloaders(
                r_band=50, g_band=30, b_band=20,
                train_dir="data/processed/train",
                test_dir="data/processed/test",
                transform=some_transform,
                batch_size=32,
                num_workers=4)
    """
    
    if use_memory:
        # Use in-memory hypercubes for faster band extraction
        global TRAIN_HYPERCUBES, TEST_HYPERCUBES, CLASS_NAMES
        
        # Load hypercubes into memory if not already loaded
        if TRAIN_HYPERCUBES is None or TEST_HYPERCUBES is None:
            load_hypercubes_to_memory(train_dir, test_dir)
        
        # Generate RGB arrays from in-memory hypercubes
        train_rgb_arrays, train_labels = generate_rgb_from_memory(
            TRAIN_HYPERCUBES, r_band, g_band, b_band
        )
        test_rgb_arrays, test_labels = generate_rgb_from_memory(
            TEST_HYPERCUBES, r_band, g_band, b_band
        )
        class_names = CLASS_NAMES
        
    else:
        # Load numpy arrays and create RGB images (original method)
        train_rgb_arrays, train_labels, class_names = load_numpy_arrays_to_rgb(
            train_dir, r_band, g_band, b_band
        )
        test_rgb_arrays, test_labels, _ = load_numpy_arrays_to_rgb(
            test_dir, r_band, g_band, b_band
        )
    
    # Create datasets using MemoryDataset
    train_data = MemoryDataset(train_rgb_arrays, train_labels, transform=transform)
    test_data = MemoryDataset(test_rgb_arrays, test_labels, transform=transform)
    
    # Create data loaders
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


def create_dataloaders_fast(
    r_band: int,
    g_band: int,
    b_band: int,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):
    """Fast dataloader creation using pre-loaded in-memory hypercubes.
    
    This function assumes hypercubes are already loaded in memory via load_hypercubes_to_memory().
    Use this for genetic algorithm optimization where you need to quickly test many band combinations.
    
    Args:
        r_band: Red channel band index.
        g_band: Green channel band index.
        b_band: Blue channel band index.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
    """
    global TRAIN_HYPERCUBES, TEST_HYPERCUBES, CLASS_NAMES
    
    if TRAIN_HYPERCUBES is None or TEST_HYPERCUBES is None:
        raise RuntimeError("Hypercubes not loaded in memory. Call load_hypercubes_to_memory() first.")
    
    # Generate RGB arrays from in-memory hypercubes (very fast)
    train_rgb_arrays, train_labels = generate_rgb_from_memory(
        TRAIN_HYPERCUBES, r_band, g_band, b_band
    )
    test_rgb_arrays, test_labels = generate_rgb_from_memory(
        TEST_HYPERCUBES, r_band, g_band, b_band
    )
    
    # Create datasets using MemoryDataset
    train_data = MemoryDataset(train_rgb_arrays, train_labels, transform=transform)
    test_data = MemoryDataset(test_rgb_arrays, test_labels, transform=transform)
    
    # Create data loaders
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

    return train_dataloader, test_dataloader, CLASS_NAMES


def load_numpy_arrays_to_rgb(data_dir: str, r_band: int, g_band: int, b_band: int):
    """
    Load numpy arrays from class directories and convert to RGB using specified bands.
    
    Args:
        data_dir: Path to directory containing class subdirectories with .npy files
        r_band: Red channel band index
        g_band: Green channel band index  
        b_band: Blue channel band index
        
    Returns:
        tuple: (rgb_arrays, labels, class_names)
            rgb_arrays: List of RGB numpy arrays (H, W, 3)
            labels: List of corresponding class indices
            class_names: Sorted list of class names
    """
    rgb_arrays = []
    labels = []
    
    # Get class directories and sort them
    class_names = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    class_names.sort()
    
    print(f"🎨 Loading numpy arrays from {data_dir}")
    print(f"📊 Classes found: {class_names}")
    print(f"🌈 Using RGB bands: R={r_band}, G={g_band}, B={b_band}")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        
        # Get all .npy files in the class directory
        npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        print(f"📂 Loading {len(npy_files)} files from class {class_name}")
        
        for npy_file in npy_files:
            file_path = os.path.join(class_dir, npy_file)
            
            # Load hyperspectral patch (already corrected and normalized)
            hypercube_patch = np.load(file_path)
            
            # Extract RGB bands and create RGB image
            rgb_image = create_rgb_from_patch(hypercube_patch, r_band, g_band, b_band)
            
            # Data is already corrected and normalized to [0, 255] as uint8
            # No additional normalization needed
            if rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)
            
            rgb_arrays.append(rgb_image)
            labels.append(class_idx)
    
    print(f"✅ Loaded {len(rgb_arrays)} RGB arrays")
    
    return rgb_arrays, labels, class_names


class MemoryDataset(Dataset):
    """Custom Dataset that holds RGB arrays and labels in memory"""

    def __init__(self, rgb_arrays, labels, transform=None):
        """
        Args:
            rgb_arrays: List of RGB numpy arrays (H, W, 3)
            labels: List of corresponding class labels (C0, C1, C2, C3)
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


def create_rgb_from_patch(patch, r, g, b):
    """Create RGB image from hyperspectral patch using specified bands.
    
    Args:
        patch: Hyperspectral patch (H, W, bands) - already corrected and normalized
        r, g, b: Band indices for RGB channels
        
    Returns:
        RGB image (H, W, 3) as uint8
    """
    rgb = patch[:, :, [r, g, b]]
    # Data is already radiometrically corrected and normalized to [0, 255] as uint8
    return rgb