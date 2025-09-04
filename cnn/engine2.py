"""
Contains functions for training and testing a PyTorch model with comprehensive evaluation metrics.
"""

import torch
from typing import Dict, List, Optional
from torch.amp import GradScaler  # type: ignore
from torcheval.metrics.functional import (
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
    multiclass_confusion_matrix,
)
from sklearn.metrics import classification_report
import warnings


def calculate_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes in the dataset
    Returns:
        Dictionary with accuracy, precision, recall, and f1 scores.
    """
    accuracy = (y_pred == y_true).float().mean().item()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = multiclass_precision(
            y_true, y_pred, num_classes=num_classes, average="macro"
        )
        recall = multiclass_recall(
            y_true, y_pred, num_classes=num_classes, average="macro"
        )
        f1 = multiclass_f1_score(
            y_true, y_pred, num_classes=num_classes, average="macro"
        )
    return {
        "accuracy": accuracy,
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    scaler: Optional[torch.amp.GradScaler] = None, # type: ignore
) -> Dict[str, float]:
    """Trains a PyTorch model for a single epoch using Automatic Mixed Precision (AMP).

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step) using mixed precision
    if a CUDA device is used.

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        num_classes: Number of classes in the dataset.
        scaler: An optional GradScaler for mixed precision training.

    Returns:
        A dictionary of training metrics including loss, accuracy, precision, recall, and f1.
    """
    # Put model in train mode
    model.train()

    # Setup training metrics
    train_loss = 0
    all_preds = []
    all_labels = []

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass with autocast
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
        ):
            y_pred = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward with scaler
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 5. Optimizer step with scaler
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Collect predictions and labels for metric calculation
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        all_preds.extend(y_pred_class.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    # Calculate average loss
    train_loss = train_loss / len(dataloader)

    # Calculate comprehensive metrics
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    metrics = calculate_metrics(all_labels, all_preds, num_classes)

    # Add loss to metrics
    metrics["loss"] = train_loss

    return metrics


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        num_classes: Number of classes in the dataset.

    Returns:
        A dictionary of testing metrics including loss, accuracy, precision, recall, and f1.
    """
    # Put model in eval mode
    model.eval()

    # Setup test metrics
    test_loss = 0
    all_preds = []
    all_labels = []

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Collect predictions and labels for metric calculation
            test_pred_labels = test_pred_logits.argmax(dim=1)
            all_preds.extend(test_pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Calculate average loss
    test_loss = test_loss / len(dataloader)

    # Calculate comprehensive metrics
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    metrics = calculate_metrics(all_labels, all_preds, num_classes)

    # Add loss to metrics
    metrics["loss"] = test_loss

    return metrics


def print_classification_report(
    y_true: torch.Tensor, y_pred: torch.Tensor, class_names: Optional[List[str]] = None
):
    """Print detailed classification report using sklearn.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names for better readability
    """

    # Convert to numpy for sklearn
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    print(
        classification_report(y_true_np, y_pred_np, target_names=class_names, digits=4)
    )


def print_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
):
    """Print confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        class_names: Optional list of class names for better readability
    """
    
    # Calculate confusion matrix using torcheval
    cm = multiclass_confusion_matrix(y_pred, y_true, num_classes=num_classes)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Print header
    print(f"{'':>12}", end="")
    for name in class_names:
        print(f"{name:>10}", end="")
    print("\nPredicted")

    # Print matrix with labels
    for i, name in enumerate(class_names):
        print(f"{name:>10}", end="  ")
        for j in range(num_classes):
            print(f"{cm[i, j].item():>8}", end="  ")
        print()
    print()


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    num_classes: int,
    verbose: bool = True,
    device: torch.device = torch.device("cuda"),
    class_names: Optional[List[str]] = None,
    print_final_report: bool = True,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model with comprehensive evaluation metrics.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        num_classes: Number of classes in the dataset.
        verbose: Whether to print training progress.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        class_names: Optional list of class names for better readability in reports.
        print_final_report: Whether to print detailed classification report at the end.

    Returns:
        A dictionary of training and testing metrics. Each metric has a value in a list for
        each epoch. Includes loss, accuracy, precision, recall, and f1 for both train and test.
    """
    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "test_loss": [],
        "test_acc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
    }

    # Make sure model on target device
    model.to(device)

    # Initialize GradScaler if device is CUDA
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    # Keep track of best test accuracy for final report
    best_test_acc = 0
    best_epoch_preds = None
    best_epoch_labels = None

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        # Training step
        train_metrics = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            scaler=scaler,
        )

        # Testing step
        test_metrics = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            num_classes=num_classes,
        )

        # Print progress if verbose
        if verbose:
            print(
                f"Epoch: {epoch + 1:03d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Train F1: {train_metrics['f1']:.4f} | "
                f"Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | "
                f"Test F1: {test_metrics['f1']:.4f}"
            )

        # Update results dictionary
        results["train_loss"].append(train_metrics["loss"])
        results["train_acc"].append(train_metrics["accuracy"])
        results["train_precision"].append(train_metrics["precision"])
        results["train_recall"].append(train_metrics["recall"])
        results["train_f1"].append(train_metrics["f1"])

        results["test_loss"].append(test_metrics["loss"])
        results["test_acc"].append(test_metrics["accuracy"])
        results["test_precision"].append(test_metrics["precision"])
        results["test_recall"].append(test_metrics["recall"])
        results["test_f1"].append(test_metrics["f1"])

        # Keep track of best performing epoch for final report
        if test_metrics["accuracy"] > best_test_acc:
            best_test_acc = test_metrics["accuracy"]
            # Store predictions from best epoch for final report
            if print_final_report:
                model.eval()
                all_preds = []
                all_labels = []
                with torch.inference_mode():
                    for X, y in test_dataloader:
                        X, y = X.to(device), y.to(device)
                        pred_logits = model(X)
                        pred_labels = pred_logits.argmax(dim=1)
                        all_preds.extend(pred_labels.cpu().numpy())
                        all_labels.extend(y.cpu().numpy())
                best_epoch_preds = torch.tensor(all_preds)
                best_epoch_labels = torch.tensor(all_labels)

    # Print final detailed report
    if print_final_report and best_epoch_preds is not None:
        print(f"\n{'=' * 60}")
        print(
            f"TRAINING COMPLETED - BEST MODEL PERFORMANCE (Test Acc: {best_test_acc:.4f})"
        )
        print(f"{'=' * 60}")

        if best_epoch_labels is not None and best_epoch_preds is not None:
            print_classification_report(
                best_epoch_labels, best_epoch_preds, class_names
            )
        if best_epoch_labels is not None and best_epoch_preds is not None:
            print_confusion_matrix(
                best_epoch_labels, best_epoch_preds, num_classes, class_names
            )

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return the filled results at the end of the epochs
    return results
