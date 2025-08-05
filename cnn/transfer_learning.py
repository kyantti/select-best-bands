"""
Intoxicated Fresh Figs Classification Script

This script demonstrates transfer learning using a pretrained ResNet50 model
for classifying fresh figs by toxin level: C0 (healthy), C1 (low toxin), C2 (medium toxin), C3 (high toxin).
"""

import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from timeit import default_timer as timer
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from PIL import Image
from torchinfo import summary
from .going_modular import data_setup, engine
from .helper_functions import plot_loss_curves


def setup_device():
    """Setup device agnostic code"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def create_transforms():
    """Create transforms for the data optimized for fig classification"""
    # Manual transforms with data augmentation for better generalization
    manual_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Test transforms (no augmentation for inference)
    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Auto transforms using ResNet50 weights
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    auto_transforms = weights.transforms()

    return manual_transforms, test_transforms, auto_transforms, weights


def setup_model(weights, class_names, device):
    """Setup pretrained ResNet50 model with custom classifier for fig toxin classification"""
    # Setup the model with pretrained weights and send it to the target device
    model = torchvision.models.resnet50(weights=weights).to(device)

    # Freeze all base layers except the final few for fine-tuning
    # For better feature extraction, we'll freeze early layers but allow later layers to adapt
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Get the number of input features for the classifier
    num_features = model.fc.in_features

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Replace the classifier layer with a custom one for fig toxin classification
    model.fc = torch.nn.Linear(num_features, output_shape).to(device)

    return model


def train_fig_toxin_classifier(
    train_dir: Path = Path("data/figs/train"), test_dir: Path = Path("data/figs/test")
) -> Tuple[nn.Module, dict]:
    """Train a ResNet50 model for fig toxin classification using transfer learning"""
    print("Starting Fig Toxin Classification Pipeline...")

    # Check PyTorch and torchvision versions
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

    # Setup device
    device = setup_device()

    # Create transforms
    manual_transforms, test_transforms, auto_transforms, weights = create_transforms()

    # Create DataLoaders (using auto transforms for training and test transforms for validation)
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=str(train_dir),
        test_dir=str(test_dir),
        transform=manual_transforms,  # Use manual transforms with data augmentation for training
        batch_size=32,
    )

    print(f"Class names: {class_names}")
    print("Expected classes: ['C0', 'C1', 'C2', 'C3'] (healthy to high toxin)")
    print(f"Train dataloader: {len(train_dataloader)} batches")
    print(f"Test dataloader: {len(test_dataloader)} batches")

    # Setup model
    model = setup_model(weights, class_names, device)

    # Print model summary
    print("\nModel Summary:")
    summary(
        model=model,
        input_size=(32, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )

    # Define loss and optimizer optimized for fine-grained classification
    loss_fn = nn.CrossEntropyLoss()
    # Use a lower learning rate for fine-tuning and add weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Set the random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Start the timer
    start_time = timer()

    # Setup training and save the results
    print("\nStarting training...")
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=100,  # Maximum epochs for best convergence on toxin classification
        device=device,
    )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

    # Save the trained model
    model_save_path = Path("cnn/models/fig_toxin_resnet50_100epochs.pth")
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

    # Print final performance summary
    final_train_acc = results["train_acc"][-1]
    final_test_acc = results["test_acc"][-1]
    print(f"\n[RESULTS] Final Training Accuracy: {final_train_acc:.2%}")
    print(f"[RESULTS] Final Test Accuracy: {final_test_acc:.2%}")
    print(f"[RESULTS] Best Test Accuracy: {max(results['test_acc']):.2%}")

    # Plot the loss curves
    print("\nPlotting loss curves...")
    plot_loss_curves(results)
    plt.show()

    return model, results


def evaluate_band_combination_fitness(hypercubes, r_band, g_band, b_band, 
                                    epochs=30, batch_size=32, learning_rate=0.0001,
                                    random_seed=42, verbose=False):
    """
    Evaluate fitness of a band combination by training ResNet50 and returning test accuracy.
    
    Args:
        hypercubes: Pre-loaded hypercube data
        r_band, g_band, b_band: Band indices to evaluate
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer (lower for transfer learning)
        random_seed: Random seed for reproducible results
        verbose: Whether to print training progress
        
    Returns:
        float: Test accuracy (fitness score)
    """
    try:
        # Import necessary modules
        import torch
        import torch.nn as nn
        import torchvision
        from torchvision import transforms
        from .going_modular.data_setup import create_memory_dataloaders, generate_rgb_arrays_from_bands, MemoryDataset
        from .going_modular.engine import train_step, test_step
        from torch.utils.data import DataLoader
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        # Create transforms optimized for ResNet50
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet50 input size
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Create dataloaders for this band combination
        train_dl, test_dl, class_names = create_memory_dataloaders(
            hypercubes, r_band, g_band, b_band, 
            batch_size=batch_size, random_seed=random_seed,
            transform=train_transforms, image_size=(224, 224)
        )
        
        # Create test dataloader with test transforms
        rgb_arrays, labels, _ = generate_rgb_arrays_from_bands(
            hypercubes, r_band, g_band, b_band
        )
        
        # Create train/test split (same as in create_memory_dataloaders)
        import numpy as np
        np.random.seed(random_seed)
        indices = np.random.permutation(len(rgb_arrays))
        split_idx = int(len(rgb_arrays) * 0.8)
        test_indices = indices[split_idx:]
        
        test_rgb = [rgb_arrays[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        test_dataset = MemoryDataset(test_rgb, test_labels, test_transforms)
        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
        
        if verbose:
            print(f"🎨 Evaluating bands R:{r_band}, G:{g_band}, B:{b_band}")
            print(f"   📊 Classes: {class_names}")
            print(f"   🔢 Train batches: {len(train_dl)}, Test batches: {len(test_dl)}")
        
        # Setup ResNet50 model
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights).to(device)
        
        # Freeze early layers, allow layer4 and fc to be trainable
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        
        # Replace the classifier layer
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, len(class_names)).to(device)
        
        # Setup loss function and optimizer (optimized for transfer learning)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Train model with reduced verbosity for GA
        if verbose:
            # Import the train function from going_modular
            from .going_modular.engine import train
            results = train(
                model=model,
                train_dataloader=train_dl,
                test_dataloader=test_dl,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=epochs,
                device=device
            )
        else:
            # Silent training for GA (no progress bars or prints)
            results = train_silent_resnet(
                model=model,
                train_dataloader=train_dl,
                test_dataloader=test_dl,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=epochs,
                device=device
            )
        
        # Return final test accuracy as fitness
        final_test_acc = results['test_acc'][-1]
        
        if verbose:
            print(f"   🎯 Final test accuracy: {final_test_acc:.4f}")
        
        # Clear GPU cache to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return final_test_acc
        
    except Exception as e:
        print(f"❌ Error evaluating bands R:{r_band}, G:{g_band}, B:{b_band}: {e}")
        # Clear GPU cache on error too
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0.0  # Return poor fitness for failed evaluations


def train_silent_resnet(model: torch.nn.Module, 
                       train_dataloader: torch.utils.data.DataLoader, 
                       test_dataloader: torch.utils.data.DataLoader, 
                       optimizer: torch.optim.Optimizer,
                       loss_fn: torch.nn.Module,
                       epochs: int,
                       device: torch.device) -> dict:
    """
    Silent version of train function optimized for ResNet50 (no prints or progress bars).
    """
    from .going_modular.engine import train_step, test_step
    
    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Update results dictionary (no printing)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


if __name__ == "__main__":
    model, results = train_fig_toxin_classifier()
