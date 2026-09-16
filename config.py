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

# How the train/validation boundary is drawn inside train.  The default is the
# 10 Sep boundary and reproduces it exactly; `crop_count_balanced` instead picks
# the admissible subset of train acquisitions whose crops are spread most evenly
# over the classes, because the fitness is weighted F1 and the 10 Sep validation
# weighted C0 and C2 2.5x more than C1 and C3.  Either way the train/test
# boundary, the grouping by acquisition and one acquisition per class on each
# side are untouched.
VALIDATION_BALANCE = "acquisition_stratified"
VALIDATION_BALANCES = ("acquisition_stratified", "crop_count_balanced")
# Frozen criterion, not a threshold to tune: the balanced policy publishes the
# best subset it found either way and records the ratio it reached.
VALIDATION_BALANCE_TARGET_RATIO = 1.5

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

# --- Data sanity check ----------------------------------------------------
# `check_data.py` alone: a one-off verification and a figure, never a result.
SANITY_CHECK_DIR = ROOT / "sanity-check"
SANITY_CHECK_CROPS_PER_CLASS = 2
# A copy of the published winner, so that the bare command draws the three
# channels the reported model was trained on.  The authority on which triplet a
# search chose is its own `exp_NN_ga_summary.json`, which is where
# `train_final.py` reads it from; this is a display default and nothing reads a
# result through it.  `check_data.py --bands R G B` overrides it.
SANITY_CHECK_BANDS = (366, 262, 225)
