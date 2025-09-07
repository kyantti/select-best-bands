# 🧬 Hyperspectral Band Selection with Genetic Algorithm

This project implements a genetic algorithm-based approach to optimize RGB band selection from hyperspectral imaging data for toxin classification in figs. The system combines evolutionary optimization (DEAP) with deep learning (PyTorch ResNet50 transfer learning) to identify the most informative spectral bands for distinguishing between different toxin contamination levels.

## 🔬 Research Context

Hyperspectral imaging captures data across hundreds of spectral bands, but many are redundant for specific classification tasks. This project addresses the challenge of selecting optimal band combinations that maximize classification accuracy while reducing computational complexity. The application focuses on detecting aflatoxin contamination in figs, with four classification levels: healthy (C0), low toxin (C1), medium toxin (C2), and high toxin (C3).

![Data Sanity Check](sanity-check/data_sanity_check.png)

## 📁 Project Structure

```
select-best-bands/
├── ga.py                      # Main genetic algorithm script
├── check_data.py              # Data validation and sanity checks
├── cnn/
│   ├── data_setup.py          # Data loading and preprocessing
│   ├── engine.py              # Training and evaluation utilities
│   ├── train.py               # CNN training script
│   ├── util/
│   │   └── helper_functions.py# Visualization and utilities
│   └── models/                # Saved model checkpoints directory
├── data/
│   ├── interim/
│   │   └── cropped-hypercubes/# Raw hyperspectral patches (C0-C3)
│   └── processed/
│       ├── train/             # Training data (C0-C3)
│       └── test/              # Test data (C0-C3)
├── sanity-check/
│   └── data_sanity_check.png  # Data visualization output
├── train_dataset.csv          # Training data manifest
├── test_dataset.csv           # Test data manifest
├── pyproject.toml             # Project dependencies and configuration
├── uv.lock                    # Dependency lock file
└── README.md                  # Project documentation
```

## ✨ Features

- **Genetic Algorithm Optimization:** Uses DEAP framework to evolve optimal RGB band combinations from 448 hyperspectral bands
- **Transfer Learning:** ResNet50 pre-trained model fine-tuned for toxin classification
- **Comprehensive Evaluation:** Tracks accuracy, precision, recall, F1-score, and confusion matrices
- **Memory Optimization:** Efficient data loading strategy for large hyperspectral datasets
- **Experiment Management:** Support for multiple experimental runs with detailed logging
- **Modular Architecture:** Clean separation between GA optimization and CNN training components

## 🔧 Prerequisites

- Python 3.13+
- CUDA-capable GPU (recommended for faster training)
- At least 8GB RAM for data processing
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## ⚙️ Installation

### Option 1: Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd select-best-bands

# Install dependencies
uv sync
```

### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd select-best-bands

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## 📦 Dependencies

Core packages automatically installed:
- **PyTorch Ecosystem:** torch, torchvision, torcheval, torchinfo
- **Genetic Algorithm:** deap
- **Data Science:** numpy, pandas, matplotlib, scikit-learn
- **Utilities:** tqdm, pillow, requests

## 🚀 Quick Start

### 1. Prepare Your Data

Ensure your hyperspectral data follows this structure:
- CSV files (`train_dataset.csv`, `test_dataset.csv`) with paths and class labels
- `.npy` files containing hyperspectral cubes (448 bands)
- Classes: C0 (healthy), C1 (low toxin), C2 (medium toxin), C3 (high toxin)

### 2. Run Data Validation

```bash
uv run check_data.py
```

### 3. Start Genetic Algorithm Optimization

```bash
# Interactive mode
uv run ga.py

# Background execution with logging
nohup uv run python -u ga.py > out/logs/experiment_1.log 2>&1 &
```

## 🏃 Usage

### Genetic Algorithm for Band Selection

Run the main optimization process:

```bash
uv run ga.py
```

### Background Execution

For long-running experiments:

```bash
nohup uv run python -u ga.py > out/logs/experiment_1.log 2>&1 &
```

### Direct CNN Training

Train the CNN with manually selected bands:

```bash
uv run cnn/train.py
```

### Data Validation

Check data integrity and visualize distributions:

```bash
uv run check_data.py
```

## ⚡ Configuration

### GA Parameters (in `ga.py`)

```python
# Population and Evolution
POPULATION_SIZE = 20        # Size of each generation
GENERATIONS = 50           # Number of evolutionary cycles
CROSSOVER_PROB = 0.8       # Probability of crossover
MUTATION_PROB = 0.15       # Probability of mutation
ELITISM_SIZE = 1           # Number of best individuals preserved

# CNN Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Hyperspectral Bands
START_BAND = 0             # First band index
END_BAND = 447             # Last band index (448 total bands)
```

### Customization

Modify these parameters based on your:
- **Hardware limitations:** Reduce `BATCH_SIZE` for limited GPU memory
- **Time constraints:** Adjust `GENERATIONS` and `NUM_EPOCHS`
- **Data characteristics:** Change band range for different sensors

## 🗂️ Data Format

### CSV Structure
- **Files:** `train_dataset.csv`, `test_dataset.csv`
- **Columns:** File path, class label
- **Example:**
  ```csv
  data/processed/train/C0/sample_001.npy,C0
  data/processed/train/C1/sample_002.npy,C1
  ```

### Hyperspectral Data
- **Format:** NumPy arrays (`.npy` files)
- **Shape:** `(height, width, 448)` - 448 spectral bands
- **Data Type:** Float32, normalized pixel intensities
- **Classes:** 
  - **C0:** Healthy figs (no toxin contamination)
  - **C1:** Low toxin contamination
  - **C2:** Medium toxin contamination  
  - **C3:** High toxin contamination

## 🧩 Algorithm Workflow

### 1. Initialization
- Generate random population of band combinations (triplets for RGB)
- Each individual represents 3 band indices from 448 available bands

### 2. Fitness Evaluation
- Convert hyperspectral data to RGB using selected bands
- Train ResNet50 with transfer learning on converted images
- Use test accuracy as fitness score

### 3. Evolution Process
- **Selection:** Tournament selection of parent individuals
- **Crossover:** Uniform crossover to create offspring
- **Mutation:** Random band replacement with low probability
- **Elitism:** Preserve best individuals across generations

### 4. Convergence
- Continue evolution for specified generations
- Track and save best-performing band combinations
- Output optimal RGB mapping and classification metrics

## 📊 Expected Results

### Output Files
- **Model checkpoints:** `cnn/models/checkpoint.pt`
- **Experiment logs:** Detailed training metrics and band selections
- **Visualization:** Confusion matrices, accuracy plots, data distributions

### Performance Metrics
- **Classification Accuracy:** Typically 85-95% on test data
- **Best Band Combinations:** Optimal spectral regions for toxin detection
- **Training Time:** ~30-60 minutes per generation (GPU-dependent)

### Sample Output
```
Generation 1: Best fitness = 0.847
Best bands: [156, 234, 389]
Accuracy: 84.7%, Precision: 0.851, Recall: 0.843, F1: 0.847

Generation 50: Best fitness = 0.923  
Best bands: [145, 267, 401]
Final Accuracy: 92.3%, Precision: 0.925, Recall: 0.921, F1: 0.923
```

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in ga.py
BATCH_SIZE = 16  # or smaller
```

**Data Loading Errors**
```bash
# Verify data paths and run validation
uv run check_data.py
```

**Long Training Times**
```bash
# Reduce population size and generations
POPULATION_SIZE = 10
GENERATIONS = 25
```

### System Requirements

**Minimum:**
- 8GB RAM
- Modern CPU with 4+ cores
- 5GB disk space

**Recommended:**
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- SSD storage for faster data loading

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

## 📋 License

This project is part of ongoing research. Please contact the authors for usage permissions and cite appropriately in academic work.

## 📬 Contact

- **Repository:** [GitHub Repository](https://github.com/kyantti/select-best-bands)
- **Issues:** Report bugs and feature requests via GitHub Issues
- **Branch:** Currently on `feature/band-optimization`

For research collaboration or technical questions, please open an issue with detailed information.

