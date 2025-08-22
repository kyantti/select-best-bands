"""
Intoxicated Fresh Figs Classification Script

This script demonstrates transfer learning using a pretrained ResNet50 model
for classifying fresh figs by toxin level: C0 (healthy), C1 (low toxin), C2 (medium toxin), C3 (high toxin).
"""

import torch
import torchvision
from torch import nn
from torchvision import transforms
from .util.engine import train
from .util.data_setup import create_dataloaders_fast


def setup_device():
    """Setup device agnostic code"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def eval(
    r_band,
    g_band,
    b_band,
    epochs = 30,
    batch_size = 256,
    learning_rate = 0.001
):
    """
    Fast evaluation of fitness using pre-loaded in-memory hypercubes.

    Args:
        r_band, g_band, b_band: Band indices to evaluate
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        random_seed: Random seed for reproducible results
        verbose: Whether to print training progress

    Returns:
        float: Test accuracy (fitness value)
    """

    random_seed = 42

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Create transforms optimized for ResNet50
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # ResNet50 input size
            transforms.RandomHorizontalFlip(p=0.5),
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
        num_workers=0,  # Use 0 to avoid multiprocessing issues during GA
    )

    # Setup ResNet50 model
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights).to(device)

    # Freeze early layers for faster training
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Replace the classifier layer with flatten, dense, and dropout layers
    num_features = model.fc.in_features
    num_classes = len(class_names) if class_names else 4  # Default to 4 classes
    
    # Create a new classifier with dense and dropout layers
    # Note: ResNet already has global average pooling, so we don't need Flatten
    model.fc = nn.Sequential(  # type: ignore
        nn.Linear(num_features, 512),  # Dense layer
        nn.ReLU(),
        nn.Dropout(0.5),  # Dropout layer
        nn.Linear(512, num_classes)  # Final classification layer
    ).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    results = train(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
    )

    # Save test acc - get the final epoch's test accuracy
    test_acc = results["test_acc"][-1]  # Get the last epoch's test accuracy

    # Clear GPU cache to prevent memory issues
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_acc
