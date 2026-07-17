"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
import engine
import data_setup
import torchvision
import gc
from torchinfo import summary


# Setup hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Setup band indices
BAND_INDICES = [226, 101, 369]

# Setup directories
train_csv = "train_dataset.csv"
test_csv = "test_dataset.csv"

# Setup target device with id
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Create transforms
train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(degrees=15),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# Create DataLoaders with help from data_setup.py
train_dataloader = data_setup.create_train_dataloader(
    train_csv=train_csv,
    band_indices=BAND_INDICES,
    transform=train_transform,
    batch_size=BATCH_SIZE,
    num_workers=0,
)

test_dataloader = data_setup.create_test_dataloader(
    test_csv=test_csv,
    band_indices=BAND_INDICES,
    transform=test_transform,
    batch_size=BATCH_SIZE,
    num_workers=0,
)

# Get the length of class_names (one output unit for each class)
output_shape = 4

# Model setup
weights = torchvision.models.ResNet50_Weights.DEFAULT
model = torchvision.models.resnet50(weights=weights).to(device)

# Freeze all layers except the 2 head
for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get the number of features from the last layer
in_features = model.fc.in_features

# Replace the final classifier with a simple linear layer for 4 classes
model.fc = torch.nn.Linear(in_features=in_features, out_features=4, bias=True).to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
summary(
    model,
    input_size=(
        BATCH_SIZE,
        3,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ),  # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
    verbose=1,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

# After training finishes
del model
del optimizer
del loss_fn
del train_dataloader
del test_dataloader

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()