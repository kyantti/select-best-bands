import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Ensure this import path is correct for your project structure
# For example, if this script is in the root, and your dataset class is in src/
from cnn.data_setup import HypercubeDataset


def check_dataset(csv_path, band_indices, output_dir, num_images=8):
    """Loads a few images from the dataset and saves them as a single PNG file."""
    print("--- Running Data Sanity Check ---")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # We use a simplified transform pipeline WITHOUT normalization for easier viewing
    display_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((224, 448))]
    )

    # Load the dataset
    try:
        dataset = HypercubeDataset(
            csv_file=csv_path, band_indices=band_indices, transform=display_transform
        )
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at '{csv_path}'. Please check the path.")
        return
    except Exception as e:
        print(f"❌ An error occurred while loading the dataset: {e}")
        return

    print(f"✅ Dataset loaded successfully. Total samples: {len(dataset)}")

    # Group samples by class
    class_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    num_classes = len(class_indices)
    print(f"📊 Found {num_classes} classes: {list(class_indices.keys())}")
    
    # Calculate samples per class
    samples_per_class = num_images // num_classes
    if samples_per_class == 0:
        samples_per_class = 1
        num_images = num_classes
        print(f"⚠️ Warning: Reduced num_images to {num_images} to show at least 1 sample per class.")
    
    print(f"📋 Showing {samples_per_class} samples per class ({samples_per_class * num_classes} total)")
    
    # Select balanced samples from each class
    selected_indices = []
    for label in sorted(class_indices.keys()):
        available_indices = class_indices[label]
        if len(available_indices) < samples_per_class:
            print(f"⚠️ Warning: Class {label} has only {len(available_indices)} samples, using all of them.")
            selected_indices.extend(available_indices)
        else:
            # Randomly select samples_per_class indices from this class
            perm = torch.randperm(len(available_indices))[:samples_per_class]
            selected_indices.extend([available_indices[i] for i in perm])
    
    # Update num_images to match actual selected samples
    num_images = len(selected_indices)
    
    # Adjust subplot grid
    ncols = 4
    nrows = (num_images + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()  # Flatten to handle any grid size easily

    for i, idx in enumerate(selected_indices):
        image_tensor, label = dataset[idx]

        # Convert tensor to a viewable numpy image
        # PyTorch Tensor: (C, H, W) -> Matplotlib: (H, W, C)
        image_np = image_tensor.permute(1, 2, 0).numpy()

        # Ensure data is in a displayable range [0, 1]
        image_np = np.clip(image_np, 0, 1)

        ax = axes[i]
        ax.imshow(image_np)
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    # Hide any unused subplots
    last_used_idx = len(selected_indices) - 1 if selected_indices else -1
    for j in range(last_used_idx + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Sanity Check: Balanced samples from {csv_path}")
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust layout to make room for suptitle

    # --- Save the figure to a file ---
    save_path = os.path.join(output_dir, "data_sanity_check.png")
    plt.savefig(save_path)

    print(f"\n✅ Image grid successfully saved to: {save_path}")
    print("Please download and view this file to check your data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data sanity check script.")
    parser.add_argument(
        "--csv",
        type=str,
        default="train_dataset.csv",
        help="Path to the dataset CSV file to check.",
    )
    parser.add_argument(
        "--bands", type=str, default="50,80,120", help="Comma-separated band indices."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sanity-check",
        help="Directory to save the output images.",
    )
    args = parser.parse_args()

    band_indices = [int(b) for b in args.bands.split(",")]
    check_dataset(args.csv, band_indices, args.output_dir)
