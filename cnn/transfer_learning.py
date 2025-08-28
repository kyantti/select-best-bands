"""
Intoxicated Fresh Figs Classification Script

This script demonstrates transfer learning using a pretrained ResNet50 model
for classifying fresh figs by toxin level: C0 (healthy), C1 (low toxin), C2 (medium toxin), C3 (high toxin).
"""

import torch
import torchvision
import os
import sys
from pathlib import Path
from torch import nn
from torchvision import transforms
from cnn.util.engine import train
from cnn.util.data_setup import create_dataloaders_fast
from cnn.util.helper_functions import set_seeds
from torchinfo import summary

NUM_WORKERS = os.cpu_count() or 1  # Ensure it's always an int


def setup_device():
    """Setup device agnostic code"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def setup_model(r_band, g_band, b_band, batch_size=256):
    """
    Setup the model, data loaders, and optimizer for training.

    Args:
        r_band, g_band, b_band: Band indices to evaluate
        batch_size: Batch size for training

    Returns:
        tuple: (model, train_dataloader, test_dataloader, optimizer, loss_fn, device)
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    set_seeds()

    # Create transforms optimized for ResNet50
    train_transforms = transforms.Compose(
        [
            transforms.Resize((512, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataloaders using fast in-memory approach
    train_dl, test_dl, class_names = create_dataloaders_fast(
        r_band=r_band,
        g_band=g_band,
        b_band=b_band,
        transform=train_transforms,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,  # Use 0 to avoid multiprocessing issues during GA
    )

    # Setup model
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    output_shape = len(class_names) if class_names is not None else 4

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=output_shape,  # same number of output units as our number of classes
            bias=True,
        ),
    ).to(device)

    # Setup loss function and optimizer (only optimize the new classifier layer)
    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, train_dl, test_dl, optimizer, loss_fn, device


def train_model(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    loss_fn,
    device,
    epochs=30,
    verbose=False,
):
    """
    Train the model and return the test accuracy.

    Args:
        model: The neural network model
        train_dataloader: Training data loader
        test_dataloader: Test data loader
        optimizer: Optimizer for training
        loss_fn: Loss function
        device: Device to train on
        epochs: Number of training epochs
        verbose: Whether to print training progress

    Returns:
        float: Test accuracy (fitness value)
    """
    # Train model
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
        verbose=verbose,
    )

    # Save test acc - get the final epoch's test accuracy
    test_acc = results["test_acc"][-1]  # Get the last epoch's test accuracy

    # Clear GPU cache to prevent memory issues
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_acc


def eval(r_band, g_band, b_band, epochs=30, batch_size=256, verbose=False):
    """
    Fast evaluation of fitness using pre-loaded in-memory hypercubes.

    Args:
        r_band, g_band, b_band: Band indices to evaluate
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Whether to print training progress

    Returns:
        float: Test accuracy (fitness value)
    """
    # Setup model and data
    model, train_dl, test_dl, optimizer, loss_fn, device = setup_model(
        r_band, g_band, b_band, batch_size
    )

    # Train and get accuracy
    test_acc = train_model(
        model, train_dl, test_dl, optimizer, loss_fn, device, epochs, verbose
    )

    return test_acc
