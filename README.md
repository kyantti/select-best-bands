# 🧬 Hyperspectral Band Selection with Genetic Algorithm

This project uses genetic algorithms to find optimal RGB band combinations from hyperspectral imaging data for toxin classification in fresh figs.

## 📁 Project Structure

```
select-best-bands/
├── cnn/                            # CNN and data handling modules
│   ├── going_modular/              # Essential PyTorch components
│   │   ├── data_setup.py           # Data loading and memory management
│   │   └── engine.py               # Training/testing functions
│   ├── transfer_learning.py        # 🔥 ResNet50 fitness evaluation
│   └── helper_functions.py         # Plotting and visualization
├── data/                           # Dataset directory
│   └── interim/                    
│       └── cropped-hypercubes/     # Input hyperspectral patches
│           ├── C0/                 # Class 0 (healthy)
│           ├── C1/                 # Class 1 (low toxin)
│           ├── C2/                 # Class 2 (medium toxin)
│           └── C3/                 # Class 3 (high toxin)
├── genetic_algorithm.py            # Main GA optimization script
├── test_modules.py                 # Testing and validation
├── pyproject.toml                  # Project dependencies
└── README.md                       # This file
```

## 🔥 Core Modules

### 1. Data Loading (`cnn/going_modular/data_setup.py`)
- **`load_hypercubes_to_memory()`**: Loads all hyperspectral data into RAM
- **`MemoryDataset`**: Custom PyTorch dataset for in-memory data
- **`create_memory_dataloaders()`**: Creates train/test dataloaders
- **`generate_rgb_arrays_from_bands()`**: Converts hyperspectral to RGB

### 2. CNN Evaluation (`cnn/transfer_learning.py`)
- **`evaluate_band_combination_fitness()`**: Main fitness function for GA
- **`train_silent_resnet()`**: Silent training for GA optimization
- **ResNet50 Transfer Learning**: Pre-trained ImageNet weights with fine-tuning

### 3. Genetic Algorithm (`genetic_algorithm.py`)
- **DEAP Framework**: Population-based optimization
- **Band Selection**: Optimizes 3 bands (R, G, B) from 112 total bands
- **Fitness Evaluation**: Uses ResNet50 test accuracy as fitness score
- **GPU Optimized**: Leverages NVIDIA A100 GPUs for acceleration

## 🚀 Usage

### Run Genetic Algorithm
```bash
# Start band optimization
uv run ga.py
```

### Train CNN Standalone
```bash
# Train ResNet50 directly
uv run cnn/transfer_learning.py
```

## ⚙️ Configuration

### Genetic Algorithm Parameters
- **Population Size**: 4 (conservative for testing)
- **Generations**: 2 (testing mode)
- **Band Range**: 0-111 (112 total bands)
- **Fitness Function**: ResNet50 test accuracy

### CNN Parameters
- **Model**: ResNet50 with ImageNet pre-trained weights
- **Epochs**: 5 (testing) / 30 (production)
- **Batch Size**: 32 (optimized for A100)
- **Learning Rate**: 0.0001 (transfer learning optimized)

### Data Configuration
- **Classes**: 4 toxin levels (C0, C1, C2, C3)
- **Total Patches**: ~1,123 hyperspectral patches
- **Train/Test Split**: 80/20
- **Input Size**: 224x224 (ResNet50 standard)

## 🔬 Technical Details

### Memory-Based Optimization
- All hyperspectral data loaded into RAM once