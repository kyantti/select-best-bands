import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchinfo import summary
import torchvision.transforms as transforms
import torchvision.models as models
import os
from tqdm import tqdm

from util.datautils import HypercubeDataset

IMAGE_WIDTH = 448
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

class Trainer:
    def __init__(
        self, model, train_data, val_data, optimizer, device, save_every, save_path
    ):
        self.device = device
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data  # Add validation dataloader
        self.optimizer = optimizer
        self.save_every = save_every
        self.save_path = save_path

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        self.model.train()
        b_sz = self.train_data.batch_size
        num_steps = len(self.train_data)
        print(f"[Device {self.device}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {num_steps}")

        total_loss = 0
        for source, targets in tqdm(self.train_data, desc=f"Epoch {epoch} Training"):
            source = source.to(self.device)
            targets = targets.to(self.device)
            loss = self._run_batch(source, targets)
            total_loss += loss

        avg_epoch_loss = total_loss / num_steps
        print(f"Epoch {epoch} finished. Average Training Loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def _evaluate_epoch(self, epoch):
        self.model.eval()
        num_steps = len(self.val_data)
        total_loss = 0
        total_correct = 0
        total_examples = 0

        with torch.inference_mode():
            for source, targets in tqdm(self.val_data, desc=f"Epoch {epoch} Evaluating"):
                source = source.to(self.device)
                targets = targets.to(self.device)
                output = self.model(source)
                loss = F.cross_entropy(output, targets)
                total_loss += loss.item()

                preds = output.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
                total_examples += targets.size(0)

        avg_epoch_loss = total_loss / num_steps
        accuracy = total_correct / total_examples
        print(f"Epoch {epoch} finished. Validation Loss: {avg_epoch_loss:.4f} | Validation Accuracy: {accuracy:.4f}")
        return avg_epoch_loss, accuracy

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        path = self.save_path.replace('.pth', f'_epoch_{epoch}.pth')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(ckp, path)
        print(f"Epoch {epoch} | Training checkpoint saved at {path}")

    def train(self, max_epochs: int):
        self.model.train()
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }
        for epoch in range(max_epochs):
            train_loss = self._run_epoch(epoch)
            val_loss, val_acc = self._evaluate_epoch(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if (epoch + 1) % self.save_every == 0 or epoch == max_epochs - 1:
                self._save_checkpoint(epoch)
        print("Training complete.")
        return history


def setup_model(num_classes, checkpoint_path=None):
    # Get pre-trained model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer, which will be trainable by default
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    # If loading from a checkpoint, load the state dict
    if checkpoint_path:
        print(f"Loading model weights from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    return model


def setup_dataloaders(args, band_indices):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = HypercubeDataset(csv_file=args.train_csv, band_indices=band_indices, transform=train_transform)
    test_set = HypercubeDataset(csv_file=args.test_csv, band_indices=band_indices, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def main(args):

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    model = setup_model(args.num_classes)
    model.to(device)

    summary(
        model,
        input_size=(args.batch_size, 3, *IMAGE_SIZE),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )

    # Parse band_indices from comma-separated string to list of ints
    band_indices = [int(idx) for idx in args.band_indices.split(',')]

    train_loader, test_loader = setup_dataloaders(args, band_indices)

    # For transfer learning, explicitly optimize ONLY the final layer's parameters
    optimizer = optim.SGD(model.fc.parameters(), lr=args.learning_rate, momentum=0.9)

    trainer = Trainer(
        model,
        train_data=train_loader,
        val_data=val_loader,
        optimizer=optimizer,
        device=device,
        save_every=30,
        save_path="checkpoints/my_model.pth"
    )

    history = trainer.train(max_epochs=20)

