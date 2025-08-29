import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

_cpu_count = os.cpu_count()
NUM_WORKERS = _cpu_count if _cpu_count is not None else 0

class HypercubeDataset(Dataset):
    """
    Custom PyTorch Dataset for loading pre-processed hyperspectral data.

    This class is designed to:
    1. Read a CSV manifest file containing filepaths and labels.
    2. Pre-load all hypercube data (.npy files) into RAM for fast access during training.
    3. For a given sample, extract three specified bands to form an RGB-like image.
    4. Apply transformations (e.g., from torchvision) to the resulting image.

    It assumes the .npy files were created by your preprocessing script and
    contain NumPy arrays with a dtype of `uint8` and shape (H, W, C).
    """
    def __init__(self, csv_file, band_indices, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with 'filepath' and 'label' columns.
            band_indices (list or tuple of 3 int): The indices of the three bands
                                                   to use for the R, G, and B channels.
            transform (callable, optional): A function/transform from torchvision
                                            to be applied on a sample.
        """
        # --- 1. Store Initialization Arguments ---
        self.band_indices = band_indices
        self.transform = transform

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"The specified CSV file was not found: {csv_file}")
        if len(band_indices) != 3:
            raise ValueError("`band_indices` must be a list or tuple of exactly 3 integers.")

        # --- 2. Load Annotations and Pre-load Data into RAM ---
        annotations = pd.read_csv(csv_file)
        
        self.data_samples = []
        print(f"Initializing dataset from {csv_file}...")
        for fpath in tqdm(annotations['filepath'], desc="Loading hypercubes into RAM"):
            hypercube = np.load(fpath)
            self.data_samples.append(hypercube)
        
        self.labels = annotations['label'].tolist()
        print(f"Dataset successfully loaded. Total samples: {len(self.labels)}")

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
        rgb_image = hypercube[:, :, self.band_indices]

        # --- 3. Apply Transformations ---
        # The `transform` pipeline is crucial. `transforms.ToTensor()` will convert
        # the (H, W, C) uint8 NumPy array into a (C, H, W) float32 Torch tensor
        # and automatically scale the values from the [0, 255] range to [0.0, 1.0].
        if self.transform:
            image_tensor = self.transform(rgb_image)
        else:
            # If no transform is provided, we must manually convert to a tensor and scale.
            image_tensor = torch.from_numpy(rgb_image.transpose((2, 0, 1))).to(torch.float32) / 255.0

        return image_tensor, label


def create_dataloaders(
    train_csv: str, 
    test_csv: str,
    band_indices: list[int],
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
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
  train_data = HypercubeDataset(train_csv, band_indices, transform=transform)
  test_data = HypercubeDataset(test_csv, band_indices, transform=transform)

  # Get class names
  class_names = train_data.labels

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
