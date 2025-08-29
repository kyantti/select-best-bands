"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
import data_setup
import engine
import torchvision
import gc
from torchinfo import summary
from timeit import default_timer as timer 

# Setup hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Setup band indices
BAND_INDICES = [0, 1, 2]

# Setup directories
train_csv = "train_dataset.csv"
test_csv = "test_dataset.csv"

# Setup target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create transforms
data_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_csv=train_csv,
    test_csv=test_csv,
    band_indices=BAND_INDICES,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)


# NEW: Setup the model with pretrained weights and send it to the target device (torchvision v0.13+)
weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT # .DEFAULT = best available weights
model = torchvision.models.efficientnet_b4(weights=weights).to(device)

# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
  param.requires_grad = False

# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(
        in_features=1792,
        out_features=output_shape,  # same number of output units as our number of classes
        bias=True,
    ),
).to(device)

# # Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
summary(model, 
        input_size=(BATCH_SIZE, 3, IMAGE_HEIGHT, IMAGE_WIDTH), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=1,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

# Set loss and optimizer
# Define loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Start the timer
start_time = timer()

# Setup training and save the results
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=NUM_EPOCHS,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# After training finishes
del model
del optimizer
del loss_fn
del train_dataloader
del test_dataloader

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()