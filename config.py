"""Every constant of the experiment, in one place.

These are the values that produced the 10 Sep 2026 result (weighted F1 0.722 on
a test set that never took part in the selection, bands 366/262/225).  Changing
any of them makes new runs incomparable with that one: change them only as the
declared subject of a feature, and take a new experiment number.
"""

from pathlib import Path

# --- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
HYPERCUBES_DIR = DATA_DIR / "cropped_hypercubes"  # symlink to the 1124 NPZ crops
HYPERCUBES_MANIFEST = DATA_DIR / "cropped_hypercubes.csv"
SPECTRAL_AXES = DATA_DIR / "spectral_axes.csv"
PARTITIONS = DATA_DIR / "evaluation_partitions.csv"  # written by split_dataset.py

OUT_DIR = ROOT / "out"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"
LOGS_DIR = OUT_DIR / "logs"
MODELS_DIR = OUT_DIR / "models"

# --- Labels and device ----------------------------------------------------
CLASS_NAMES = ("C0", "C1", "C2", "C3")
NUM_CLASSES = len(CLASS_NAMES)

# A name, not a `torch.device`: `config` stays importable without torch so the
# protocol tests can run on a machine with no GPU.  Call sites wrap it.
DEVICE = "cuda"

# --- Partition ------------------------------------------------------------
# Acquisition-grouped, stratified by severity class on both boundaries.
# With this seed the split is 22 / 6 / 8 acquisitions (708 / 160 / 256 crops).
PARTITION_SEED = 20230717
TEST_PROPORTION = 0.2
VALIDATION_PROPORTION = 0.2

# --- Seeds ----------------------------------------------------------------
STUDY_SEED = 23  # seeds `random` right before the GA population is created
CANDIDATE_SEED = 1729  # goes into the per-candidate derived training seed
FINAL_SEED = 2718  # the single final-model training run

# --- Genetic algorithm ----------------------------------------------------
BANDS_PER_CANDIDATE = 3
POPULATION_SIZE = 20
GENERATIONS = 25
CROSSOVER_PROBABILITY = 0.8  # cxpb
MUTATION_PROBABILITY = 0.15  # mutpb
TOURNAMENT_SIZE = 3
BLEND_ALPHA = 0.5  # cxBlend
# Repaired gaussian mutation: sigma 20 with a per-gene probability of 0.3 is what
# keeps the population from collapsing to a single candidate, as it did in the
# 20 original studies.  Every operator is followed by a repair that guarantees
# three distinct in-range bands without reordering them.
MUTATION_MU = 0.0
MUTATION_SIGMA = 20.0
MUTATION_GENE_PROBABILITY = 0.3
HALL_OF_FAME_SIZE = 1  # observes only; the winner is re-tie-broken over the cache

# --- CNN ------------------------------------------------------------------
NUM_EPOCHS = 50  # no early stopping
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
# 8 persistent workers are part of the reproducible result: the DataLoader
# generator is split across them, so another count gives another augmentation
# stream and another fitness.
NUM_WORKERS = 8

# --- Bootstrap ------------------------------------------------------------
BOOTSTRAP_RESAMPLES = 5000  # whole acquisitions, not crops
BOOTSTRAP_SEED = 1729  # the same number as CANDIDATE_SEED in the real run, on purpose
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
# The two tails of that level, in the percent `numpy.percentile` wants.
BOOTSTRAP_PERCENTILES = (2.5, 97.5)
