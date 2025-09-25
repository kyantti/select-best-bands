"""
Contains functions for training and testing a PyTorch model.
"""

import torch

from typing import Dict, List, Tuple, Optional
from torch.amp import GradScaler  # type: ignore

# Import scikit-learn metrics
from sklearn.metrics import confusion_matrix, classification_report


class EarlyStopping:
    """Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0, restore_best_weights: bool = False):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            restore_best_weights: Whether to restore model weights from the best epoch.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Args:
            val_loss: Validation loss for the current epoch.
            model: PyTorch model being trained.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model: torch.nn.Module):
        """Save model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None, # type: ignore
) -> Tuple[float, float]:  # type: ignore
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
    scaler: An optional GradScaler for mixed precision training.

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass with autocast
        # Recommended - more explicit:
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
        ):
            y_pred = model(X)

        # 2. Calculate  and accumulate loss
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

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss, testing accuracy, all predictions, and all labels.
    In the form (test_loss, test_accuracy, y_preds, y_true). For example:

    (0.0223, 0.8985, [0, 1, 0, ...], [0, 1, 1, ...])
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Lists to store all predictions and labels
    y_preds, y_true = [], []

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

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

            # Append predictions and labels to lists
            y_preds.extend(test_pred_labels.cpu().tolist())
            y_true.extend(y.cpu().tolist())

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc, y_preds, y_true


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    verbose: bool,
    device: torch.device,
    class_names: List[str],
    early_stopping: Optional[EarlyStopping] = None,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model with optional mixed precision and early stopping.

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
    verbose: Whether to print training progress.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    class_names: A list of the target classes.
    early_stopping: An optional EarlyStopping instance to stop training early.

    Returns:
    A dictionary of training and testing loss, accuracy, and for the last epoch,
    confusion matrix and classification report.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...],
              confusion_matrix: np.ndarray,
              classification_report: dict}
    """
    # Create empty results dictionary
    results: Dict[str, List] = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Make sure model on target device
    model.to(device)

    # Initialize GradScaler if device is CUDA
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )
        test_loss, test_acc, y_preds, y_true = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        # Print out what's happening
        if verbose:
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f} | "
            )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # If it's the last epoch, calculate confusion matrix and classification report
        if epoch == epochs - 1:
            results["confusion_matrix"] = [confusion_matrix(y_true, y_preds)]
            results["classification_report"] = [classification_report(y_true, y_preds, target_names=class_names, output_dict=True)]

        # Early stopping check
        if early_stopping is not None:
            if early_stopping(test_loss, model):
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                # Calculate final metrics if stopping early
                if "confusion_matrix" not in results:
                    results["confusion_matrix"] = [confusion_matrix(y_true, y_preds)]
                    results["classification_report"] = [classification_report(y_true, y_preds, target_names=class_names, output_dict=True)]
                break

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return the filled results at the end of the epochs
    return results