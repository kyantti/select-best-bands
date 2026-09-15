"""Genetic search for the best band triplet, and the evaluation of a single one.

    uv run python ga.py N                         # the search of experiment N
    uv run python ga.py --evaluate 366 262 225    # train and score one triplet

The search ranks candidates by weighted F1 on the validation crops and never
opens a held-out test crop.  Every candidate it trains is appended to
`out/tables/exp_NN_candidates.csv` the moment it finishes, and relaunching the
same experiment number replays the same trajectory over that file: the study
seed fixes which candidates are proposed, so a candidate already in the CSV is a
free hit and the search picks up where it stopped.  No DEAP state is serialized.

The operators and the evolution loop are copied from fig-aflatoxin's genetic
module and GA stage; the cache, the outputs and the CLI are this repo's.
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import matplotlib
import torch
from deap import algorithms, base, tools

import config
from cnn.data_setup import (
    Hypercube,
    load_hypercubes,
    load_partitions,
    load_spectral_axis,
    spectral_axis_id,
    wavelengths_of,
)
from cnn.model import (
    RESNET50_WEIGHTS,
    SEED_POLICY,
    candidate_seed,
    data_identity,
    evaluate_candidate,
)

matplotlib.use("Agg")  # the search runs headless, under nohup
import matplotlib.pyplot as plt  # noqa: E402

Candidate = tuple[int, int, int]


# --- Genetic operators -----------------------------------------------------
# Copied from fig-aflatoxin's `modeling/genetic.py`.  `rng` is the `random`
# module itself, so the whole trajectory hangs off one seeded generator.


@dataclass(frozen=True)
class CandidateFitness:
    """What one band triplet scored on the validation crops.

    `weighted_f1` is the value the search ranks on; the rest are diagnostics
    that also break its ties.
    """

    weighted_f1: float
    macro_f1: float
    ordinal_mae: float
    quadratic_weighted_kappa: float

    def __post_init__(self) -> None:
        """Reject a fitness that is not a number.

        Raises:
            ValueError: If any of the four values is infinite or nan.
        """
        if not all(
            math.isfinite(value)
            for value in (
                self.weighted_f1,
                self.macro_f1,
                self.ordinal_mae,
                self.quadratic_weighted_kappa,
            )
        ):
            raise ValueError("candidate evaluator returned invalid fitness")


def validate_band_count(band_count: int) -> None:
    """Raise unless the SpectralAxis can hold a triplet at all.

    Raises:
        ValueError: If there are fewer than three bands to choose from.
    """
    if band_count < 3:
        raise ValueError("a search needs a SpectralAxis with at least three bands")


def clamp(value: int, band_count: int) -> int:
    """Pull a band index back inside the SpectralAxis."""
    return min(max(value, 0), band_count - 1)


def initialize_candidate(rng, band_count: int) -> Candidate:
    """Draw one candidate of three distinct bands from the SpectralAxis."""
    validate_band_count(band_count)
    values = rng.sample(range(band_count), 3)
    return values[0], values[1], values[2]


def crossover_candidates(
    left: Candidate, right: Candidate, rng, *, band_count: int, alpha: float
) -> tuple[Candidate, Candidate]:
    """Blend two parents and repair both children back to distinct in-range bands."""
    validate_band_count(band_count)
    children = [list(left), list(right)]
    for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
        gamma = (1 + 2 * alpha) * rng.random() - alpha
        children[0][index] = clamp(int((1 - gamma) * left_value + gamma * right_value), band_count)
        children[1][index] = clamp(int(gamma * left_value + (1 - gamma) * right_value), band_count)
    return (
        repair_duplicates(children[0], rng, band_count),
        repair_duplicates(children[1], rng, band_count),
    )


def mutate_candidate(
    candidate: Candidate, rng, *, band_count: int, mu: float, sigma: float, gene_probability: float
) -> Candidate:
    """Perturb genes by a Gaussian step and repair the result."""
    validate_band_count(band_count)
    mutated = list(candidate)
    for index, value in enumerate(mutated):
        if rng.random() < gene_probability:
            mutated[index] = clamp(value + int(rng.gauss(mu, sigma)), band_count)
    return repair_duplicates(mutated, rng, band_count)


def repair_candidate(values, rng, band_count: int) -> Candidate:
    """Clamp values into the SpectralAxis and resolve duplicates to distinct bands."""
    validate_band_count(band_count)
    return repair_duplicates(list(values), rng, band_count)


def repair_duplicates(values: list[int], rng, band_count: int) -> Candidate:
    """Replace duplicates in place, so a candidate is never reordered.

    Only the repeated position moves: the bands that were already distinct stay
    where they are, which matters because the three bands are the R, G and B
    channels of the image and swapping them is a different candidate.
    """
    used: set[int] = set()
    repaired: list[int] = []
    for position, value in enumerate(values):
        value = clamp(value, band_count)
        if value in used:
            surviving_later = {
                clamp(later, band_count) for later in values[position + 1 :] if later not in used
            }
            available = [
                index
                for index in range(band_count)
                if index not in used and index not in surviving_later
            ]
            if not available:
                available = [index for index in range(band_count) if index not in used]
            value = rng.choice(available)
        used.add(value)
        repaired.append(value)
    return repaired[0], repaired[1], repaired[2]


# --- The candidate cache ---------------------------------------------------


LIBRARIES = ("torch", "torchvision", "numpy", "scikit-learn", "deap")

CANDIDATE_FIELDS = (
    "b0",
    "b1",
    "b2",
    "candidate_seed",
    "weighted_f1",
    "macro_f1",
    "ordinal_mae",
    "quadratic_weighted_kappa",
    "first_generation",
    "seconds",
    "wavelengths_nm",
)
METRIC_FIELDS = ("weighted_f1", "macro_f1", "ordinal_mae", "quadratic_weighted_kappa")


class CandidateCache:
    """Every candidate this experiment has trained, on disk, appended as it finishes.

    The file is the experiment's memory.  A row is flushed and fsynced before the
    next candidate starts, so a run killed at any point loses at most the
    candidate that was training, and the sidecar records the contract those
    numbers were produced under so a warm file cannot be read under another one.
    """

    def __init__(
        self,
        path: Path,
        sidecar: Path,
        *,
        identity: dict[str, str],
        wavelengths: list[float],
        fitness_contract: dict,
        ga_parameters: dict,
    ) -> None:
        self.path = Path(path)
        self.sidecar = Path(sidecar)
        self.identity = identity
        self.wavelengths = wavelengths
        self.entries: dict[Candidate, CandidateFitness] = {}
        self.first_generations: dict[Candidate, int] = {}
        self.check_sidecar(fitness_contract, ga_parameters)
        self.restore()
        # Snapshotted here: `entries` keeps growing as the search trains, and the
        # question this answers is how much of the file was already on disk.
        self.restored = len(self.entries)

    def check_sidecar(self, fitness_contract: dict, ga_parameters: dict) -> None:
        """Compare this run's contract with the one the file was written under.

        A different fitness contract means the cached numbers were produced by a
        different experiment, and mixing them would be a silent lie: that is an
        error.  Different GA parameters only change which candidates come next,
        which a replay handles, so that is a warning.

        Raises:
            ValueError: If the fitness contract differs from the recorded one.
        """
        current = {"schema_version": 1, "fitness": fitness_contract, "ga": ga_parameters}
        if not self.sidecar.exists():
            self.sidecar.parent.mkdir(parents=True, exist_ok=True)
            self.sidecar.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
            return
        recorded = json.loads(self.sidecar.read_text())
        if recorded.get("fitness") != fitness_contract:
            differing = sorted(
                name
                for name in set(recorded.get("fitness", {})) | set(fitness_contract)
                if recorded.get("fitness", {}).get(name) != fitness_contract.get(name)
            )
            raise ValueError(
                f"{self.path} was written under another fitness contract "
                f"({', '.join(differing)} differ); its numbers are not comparable. "
                f"Use a new experiment number, or delete {self.sidecar} and the CSV."
            )
        if recorded.get("ga") != ga_parameters:
            print(f"WARNING: the GA parameters differ from {self.sidecar}; the trajectory changes")
            # Rewritten, or the sidecar would keep describing the previous run
            # while the statistics beside it came from this one.
            self.sidecar.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    def restore(self) -> None:
        """Read the rows already on disk, checking each one's candidate seed.

        Raises:
            ValueError: If the file has an unexpected schema, or if a row's
                recorded seed is not the one this data identity derives, which
                means the row was produced against other data.
        """
        if not self.path.exists():
            return
        with self.path.open(newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != CANDIDATE_FIELDS:
                raise ValueError(f"{self.path} has an incompatible schema")
            for row in reader:
                candidate = (int(row["b0"]), int(row["b1"]), int(row["b2"]))
                expected = candidate_seed(candidate, self.identity)
                if int(row["candidate_seed"]) != expected:
                    raise ValueError(
                        f"{self.path}: candidate {candidate} carries candidate seed "
                        f"{row['candidate_seed']}, but this data derives {expected}. "
                        "The cache was produced against other data."
                    )
                self.entries[candidate] = CandidateFitness(
                    *(float(row[name]) for name in METRIC_FIELDS)
                )
                self.first_generations[candidate] = int(row["first_generation"])

    def get(self, candidate: Candidate) -> CandidateFitness | None:
        """The recorded fitness of `candidate`, or None if it was never trained."""
        return self.entries.get(candidate)

    def append(
        self, candidate: Candidate, fitness: CandidateFitness, *, first_generation: int, seconds: float
    ) -> None:
        """Write one finished candidate through to disk before the next one starts."""
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=CANDIDATE_FIELDS, lineterminator="\n")
            if new:
                writer.writeheader()
            writer.writerow(
                {
                    "b0": candidate[0],
                    "b1": candidate[1],
                    "b2": candidate[2],
                    "candidate_seed": candidate_seed(candidate, self.identity),
                    **{name: repr(getattr(fitness, name)) for name in METRIC_FIELDS},
                    "first_generation": first_generation,
                    "seconds": repr(seconds),
                    "wavelengths_nm": json.dumps(
                        [self.wavelengths[index] for index in candidate]
                    ),
                }
            )
            target.flush()
            os.fsync(target.fileno())
        self.entries[candidate] = fitness
        self.first_generations[candidate] = first_generation


def refuse_to_evaluate(candidate: Candidate) -> CandidateFitness:
    """The evaluator of `--no-evaluate`: every cache miss is an error.

    Raises:
        LookupError: Always.
    """
    raise LookupError(f"candidate {candidate} is not in the cache and --no-evaluate was given")


# --- The search ------------------------------------------------------------


@dataclass(frozen=True)
class SearchParameters:
    """The GA's knobs, defaulting to the ones the recorded run used."""

    population_size: int = config.POPULATION_SIZE
    generations: int = config.GENERATIONS
    crossover_probability: float = config.CROSSOVER_PROBABILITY
    mutation_probability: float = config.MUTATION_PROBABILITY
    tournament_size: int = config.TOURNAMENT_SIZE
    blend_alpha: float = config.BLEND_ALPHA
    mutation_mu: float = config.MUTATION_MU
    mutation_sigma: float = config.MUTATION_SIGMA
    gene_mutation_probability: float = config.MUTATION_GENE_PROBABILITY

    def as_dict(self) -> dict:
        """The parameters as JSON, for the cache sidecar."""
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    """The winner, every fitness the search saw, and the per-generation record."""

    winner: Candidate
    fitness: dict[Candidate, CandidateFitness]  # only what this run proposed
    generations: list[dict]
    cache_hits: int
    evaluations: int


def candidate_fitness_key(item) -> tuple:
    """Order candidates deterministically by fitness, then by band indices.

    The index term only breaks ties, and it breaks them the same way whatever
    order the candidates were found in.
    """
    candidate, fitness = item
    return (
        fitness.weighted_f1,
        -fitness.ordinal_mae,
        tuple(-index for index in candidate),
    )


class WeightedF1Fitness(base.Fitness):
    """DEAP's fitness, maximizing the single weighted-F1 value."""

    weights = (1.0,)


class Individual(list):
    """A candidate as DEAP wants it: a plain list that carries a fitness.

    A subclass rather than `deap.creator`, which builds classes at import time
    into a module-level registry and makes two searches in one process collide.
    """

    def __init__(self, values=()) -> None:
        super().__init__(values)
        self.fitness = WeightedF1Fitness()


def run_search(band_count: int, evaluator, cache: CandidateCache, *, params: SearchParameters):
    """Evolve band triplets, training only the ones the cache has never seen.

    `random` must already be seeded by the caller: the seeding has to happen
    immediately before the population is drawn, and doing it here would hide
    that.  The evolution's random state is saved and restored around every call
    to `evaluator`, so a run that trains a candidate and a run that reads it
    back from the cache propose exactly the same candidates afterwards.  That is
    what makes a resume a replay rather than a different search.
    """
    generation = 0
    cache_hits = 0
    evaluations = 0
    # What this run actually proposed.  The cache can hold more than that — rows
    # left by a run with other GA parameters, or by ticket 06's preloaded replay
    # — and a winner the search never met would not be a search result.
    seen: dict[Candidate, CandidateFitness] = {}

    def as_candidate(individual) -> Candidate:
        return individual[0], individual[1], individual[2]

    def evaluate(individual) -> tuple[float]:
        nonlocal cache_hits, evaluations
        candidate = as_candidate(individual)
        recorded = cache.get(candidate)
        if recorded is not None:
            cache_hits += 1
            seen[candidate] = recorded
            return (recorded.weighted_f1,)
        started = time.perf_counter()
        evolution_random_state = random.getstate()
        try:
            fitness = evaluator(candidate)
        finally:
            random.setstate(evolution_random_state)
        evaluations += 1
        seen[candidate] = fitness
        cache.append(
            candidate,
            fitness,
            first_generation=generation,
            seconds=time.perf_counter() - started,
        )
        return (fitness.weighted_f1,)

    def mate(left, right):
        left_candidate, right_candidate = crossover_candidates(
            as_candidate(left),
            as_candidate(right),
            random,
            band_count=band_count,
            alpha=params.blend_alpha,
        )
        left[:] = left_candidate
        right[:] = right_candidate
        return left, right

    def mutate(individual):
        individual[:] = mutate_candidate(
            as_candidate(individual),
            random,
            band_count=band_count,
            mu=params.mutation_mu,
            sigma=params.mutation_sigma,
            gene_probability=params.gene_mutation_probability,
        )
        return (individual,)

    def select(individuals, count):
        nonlocal generation
        generation += 1
        return tools.selTournament(individuals, count, tournsize=params.tournament_size)

    def fitness_values(individuals) -> list[float]:
        return [individual.fitness.values[0] for individual in individuals]

    def best_candidate(individuals) -> Candidate:
        candidates = [as_candidate(individual) for individual in individuals]
        return max(
            candidates,
            key=lambda candidate: candidate_fitness_key((candidate, seen[candidate])),
        )

    toolbox = base.Toolbox()
    toolbox.register(
        "individual", tools.initIterate, Individual, lambda: initialize_candidate(random, band_count)
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", select)

    collector = tools.Statistics(lambda individual: individual)
    collector.register("population_size", len)
    collector.register(
        "unique_population_candidates",
        lambda individuals: len({as_candidate(individual) for individual in individuals}),
    )
    collector.register("avg", lambda individuals: statistics.fmean(fitness_values(individuals)))
    collector.register("std", lambda individuals: statistics.pstdev(fitness_values(individuals)))
    collector.register("min", lambda individuals: min(fitness_values(individuals)))
    collector.register("max", lambda individuals: max(fitness_values(individuals)))
    collector.register("best", best_candidate)

    population = toolbox.population(n=params.population_size)
    # The hall of fame only observes: it holds no survivor and takes part in no
    # selection, so it cannot turn into the elitism the original study had.
    hall_of_fame = tools.HallOfFame(1)
    _, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=params.crossover_probability,
        mutpb=params.mutation_probability,
        ngen=params.generations,
        stats=collector,
        halloffame=hall_of_fame,
        verbose=False,
    )

    # DEAP's hall of fame keeps whichever tying individual it met first.  The
    # reported winner is re-tie-broken over every candidate that ties it, so the
    # answer does not depend on the order DEAP happened to visit them in.
    best_weighted_f1 = hall_of_fame[0].fitness.values[0]
    winner = max(
        (item for item in seen.items() if item[1].weighted_f1 == best_weighted_f1),
        key=candidate_fitness_key,
    )[0]
    generations = [
        {
            "gen": record["gen"],
            "nevals": record["nevals"],
            "population_size": record["population_size"],
            "unique_population_candidates": record["unique_population_candidates"],
            "avg": record["avg"],
            "std": record["std"],
            "min": record["min"],
            "max": record["max"],
            "best": record["best"],
        }
        for record in logbook
    ]
    return SearchResult(winner, seen, generations, cache_hits, evaluations)


# --- One candidate, on its own ---------------------------------------------


def select_device() -> torch.device:
    """Resolve `config.DEVICE`, refusing to quietly fall back.

    `config.DEVICE` is part of what makes a number reproducible, so an
    accelerator that was asked for and is not there is an error: falling back to
    the CPU would keep running and produce a figure nobody could reproduce.

    Raises:
        ValueError: If the configured device is not available.
    """
    if config.DEVICE == "cpu":
        return torch.device("cpu")
    if config.DEVICE.startswith("cuda") and torch.cuda.is_available():
        return torch.device(config.DEVICE)
    raise ValueError(f"config.DEVICE is '{config.DEVICE}', which this machine does not have")


def load_selection_data(*, verbose: bool = True):
    """Load the crops band selection is allowed to see, and their identity.

    Held-out test rows are dropped before a single NPZ is opened, so the search
    cannot read what it will finally be scored on.  Returns the train-fit and
    validation crops, the spectral axis, and the checksums of all three.

    Raises:
        ValueError: If the manifests leak across a boundary, if an artifact is
            malformed, or if the crops were cut against another SpectralAxis.
    """
    rows = load_partitions(config.PARTITIONS)
    train_rows = [row for row in rows if row["partition"] == "train"]
    fit_rows = [row for row in train_rows if row["selection_partition"] == "train"]
    validation_rows = [row for row in train_rows if row["selection_partition"] == "validation"]
    if verbose:
        print(f"loading {len(fit_rows)} train-fit and {len(validation_rows)} validation crops")
    training = load_hypercubes(config.HYPERCUBES_MANIFEST, fit_rows, verbose=verbose)
    validation = load_hypercubes(config.HYPERCUBES_MANIFEST, validation_rows, verbose=verbose)
    wavelengths = load_spectral_axis(config.SPECTRAL_AXES)
    axis_ids = {hypercube.spectral_axis_id for hypercube in (*training, *validation)}
    if axis_ids != {spectral_axis_id(wavelengths)}:
        raise ValueError("train crops were cut against another SpectralAxis")
    identity = data_identity(training, validation, train_rows, wavelengths)
    return training, validation, wavelengths, identity


def write_predictions(path: Path, validation: list[Hypercube], predictions: list[int]) -> None:
    """Write one row per validation crop, in the order the model saw them.

    Opened exclusively, as fig-aflatoxin opens it: a file under `out/` is part
    of the thesis record, so a second run of the same triplet has to be told to
    write elsewhere rather than quietly replace the first one's numbers.

    Raises:
        FileExistsError: If `path` is already there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("cropped_hypercube_id", "acquisition_id", "actual", "predicted"),
            lineterminator="\n",
        )
        writer.writeheader()
        for hypercube, prediction in zip(validation, predictions, strict=True):
            writer.writerow(
                {
                    "cropped_hypercube_id": hypercube.cropped_hypercube_id,
                    "acquisition_id": hypercube.acquisition_id,
                    "actual": config.CLASS_NAMES[hypercube.severity_class],
                    "predicted": config.CLASS_NAMES[prediction],
                }
            )


def evaluate_command(bands, *, epochs: int, predictions_path: Path | None) -> None:
    """Train one triplet on the train-fit crops and report its validation metrics.

    Raises:
        FileExistsError: If the predictions file is already there.  Checked
            before anything is loaded, so an hour of training is never spent on
            a result that cannot be written down.
    """
    if predictions_path is None:
        r, g, b = (int(band) for band in bands)
        predictions_path = config.TABLES_DIR / f"evaluate_{r}_{g}_{b}_validation_predictions.csv"
    if predictions_path.exists():
        raise FileExistsError(
            f"{predictions_path} is already there; results under out/ are not overwritten. "
            "Move it aside, or pass --predictions with another path."
        )
    device = select_device()
    training, validation, wavelengths, identity = load_selection_data()
    nanometres = wavelengths_of(bands, wavelengths)
    derived_seed = candidate_seed(bands, identity)
    print(f"candidate      {tuple(bands)}")
    print("wavelengths_nm " + ", ".join(f"{value}" for value in nanometres))
    print(f"candidate seed {derived_seed}")
    print(f"device         {device}  |  epochs {epochs}")

    started = time.perf_counter()
    predictions, metrics, (mean, std) = evaluate_candidate(
        bands,
        training,
        validation,
        seed=derived_seed,
        device=device,
        epochs=epochs,
    )
    elapsed = time.perf_counter() - started

    print(f"mean           {list(mean)}")
    print(f"std            {list(std)}")
    for name in ("weighted_f1", "macro_f1", "ordinal_mae", "quadratic_weighted_kappa"):
        print(f"{name:<14} {metrics[name]!r}")
    print(f"elapsed_s      {elapsed:.1f}")

    write_predictions(predictions_path, validation, predictions)
    print(f"predictions    {predictions_path}")


# --- One experiment --------------------------------------------------------


def library_versions() -> dict[str, str]:
    """The versions a fitness depends on; a different one is a different number."""
    return {name: version(name) for name in LIBRARIES}


def fitness_contract(identity: dict[str, str], band_count: int, device, *, epochs: int) -> dict:
    """Everything a cached fitness depends on, so another run cannot reuse it blindly.

    If any of this changes, the numbers already in the CSV were produced by a
    different experiment and the two cannot be mixed.
    """
    return {
        "candidate_seed": config.CANDIDATE_SEED,
        "seed_policy": SEED_POLICY,
        "epochs": epochs,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "image_size": list(config.IMAGE_SIZE),
        "num_workers": config.NUM_WORKERS,
        "device_type": device.type,
        "partition_seed": config.PARTITION_SEED,
        "test_proportion": config.TEST_PROPORTION,
        "validation_proportion": config.VALIDATION_PROPORTION,
        "band_count": band_count,
        "data_identity": identity,
        "libraries": library_versions(),
        "resnet50_weights": str(RESNET50_WEIGHTS),
    }


def make_evaluator(training, validation, identity, device, *, epochs: int):
    """Wrap `evaluate_candidate` so the search trains one triplet and reports it."""

    def evaluate(candidate: Candidate) -> CandidateFitness:
        seed = candidate_seed(candidate, identity)
        started = time.perf_counter()
        _, metrics, _ = evaluate_candidate(
            candidate, training, validation, seed=seed, device=device, epochs=epochs
        )
        print(
            f"  {candidate} seed {seed} weighted_f1 {metrics['weighted_f1']:.6f} "
            f"({time.perf_counter() - started:.1f} s)",
            flush=True,
        )
        return CandidateFitness(*(metrics[name] for name in METRIC_FIELDS))

    return evaluate


def experiment_prefix(experiment: int) -> str:
    """`exp_NN`: the name every file of one experiment starts with."""
    return f"exp_{experiment:02d}"


def experiment_paths(experiment: int) -> dict[str, Path]:
    """Every file experiment `NN` owns.  No other number writes to any of them."""
    prefix = experiment_prefix(experiment)
    return {
        "candidates": config.TABLES_DIR / f"{prefix}_candidates.csv",
        "config": config.TABLES_DIR / f"{prefix}_ga_config.json",
        "stats": config.TABLES_DIR / f"{prefix}_ga_stats.csv",
        "summary": config.TABLES_DIR / f"{prefix}_ga_summary.json",
        "figure": config.FIGURES_DIR / f"{prefix}_fitness_evolution.png",
    }


def winner_results_path(experiment: int, winner: Candidate) -> Path:
    """The winner's own file, named after its bands as every v1 result was."""
    r, g, b = winner
    return config.TABLES_DIR / f"{experiment_prefix(experiment)}_cnn_results_{r}_{g}_{b}.csv"


def refuse_to_reuse_an_old_experiment_number(paths: dict[str, Path], prefix: str) -> None:
    """Refuse a number whose results some earlier protocol already owns.

    `out/` is the thesis record.  A number that has statistics but no candidate
    cache was written by a run this code cannot resume, so continuing would
    overwrite a result instead of extending one.

    Raises:
        ValueError: If that is the case.
    """
    if paths["stats"].exists() and not paths["candidates"].exists():
        raise ValueError(
            f"{paths['stats']} exists but {paths['candidates']} does not, so experiment "
            f"{prefix} belongs to an earlier run that this search cannot resume. "
            "Use a new experiment number."
        )


def write_stats(path: Path, generations: list[dict], wavelengths: list[float]) -> None:
    """One row per generation, keeping v1's column names so old readers still work."""
    fields = (
        "gen",
        "nevals",
        "population_size",
        "unique_population_candidates",
        "avg",
        "std",
        "min",
        "max",
        "best",
        "best_wavelengths_nm",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in generations:
            best = record["best"]
            writer.writerow(
                {
                    **{name: record[name] for name in fields[:8]},
                    "best": json.dumps(list(best)),
                    "best_wavelengths_nm": json.dumps(list(wavelengths_of(best, wavelengths))),
                }
            )


def write_winner(
    path: Path,
    winner: Candidate,
    fitness: CandidateFitness,
    identity: dict[str, str],
    wavelengths: list[float],
) -> None:
    """The winner's validation metrics, in band indices and in nanometres."""
    nanometres = wavelengths_of(winner, wavelengths)
    with path.open("w", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(
            (
                "b0",
                "b1",
                "b2",
                "wavelength_0_nm",
                "wavelength_1_nm",
                "wavelength_2_nm",
                *METRIC_FIELDS,
                "candidate_seed",
            )
        )
        writer.writerow(
            (
                *winner,
                *nanometres,
                *(repr(getattr(fitness, name)) for name in METRIC_FIELDS),
                candidate_seed(winner, identity),
            )
        )


def plot_fitness_evolution(
    path: Path, generations: list[dict], nanometres: tuple[float, float, float]
) -> None:
    """The curve the thesis shows: average and best weighted F1 per generation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gens = [record["gen"] for record in generations]
    figure, axes = plt.subplots(figsize=(10, 6))
    axes.plot(gens, [record["avg"] for record in generations], label="Average", color="tab:blue")
    axes.plot(
        gens,
        [record["max"] for record in generations],
        label="Best in population",
        color="tab:green",
        linestyle="--",
    )
    axes.plot(
        gens,
        [record["min"] for record in generations],
        label="Worst in population",
        color="tab:red",
        linestyle=":",
    )
    axes.set_xlabel("Generation")
    axes.set_ylabel("Validation weighted F1")
    axes.set_title(
        "Fitness evolution — winner at "
        + ", ".join(f"{value:.2f} nm" for value in nanometres)
    )
    axes.legend()
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def search_command(
    experiment: int,
    *,
    population: int,
    generations: int,
    epochs: int,
    evaluate: bool,
) -> None:
    """Run, or replay, the search of one experiment number."""
    paths = experiment_paths(experiment)
    prefix = experiment_prefix(experiment)
    refuse_to_reuse_an_old_experiment_number(paths, prefix)
    params = SearchParameters(population_size=population, generations=generations)
    device = select_device()
    training, validation, wavelengths, identity = load_selection_data()
    band_count = len(wavelengths)
    cache = CandidateCache(
        paths["candidates"],
        paths["config"],
        identity=identity,
        wavelengths=wavelengths,
        fitness_contract=fitness_contract(identity, band_count, device, epochs=epochs),
        ga_parameters=params.as_dict(),
    )
    evaluator = (
        make_evaluator(training, validation, identity, device, epochs=epochs)
        if evaluate
        else refuse_to_evaluate
    )
    print(
        f"{prefix}: {band_count} bands, population {params.population_size}, "
        f"{params.generations} generations, {epochs} epochs, restored {cache.restored}"
    )

    started = time.perf_counter()
    # Seeded here and nowhere else: the population is drawn on the next line, and
    # a seed set any later would make the first generation depend on whatever
    # loading the crops happened to consume.
    random.seed(config.STUDY_SEED)
    result = run_search(band_count, evaluator, cache, params=params)
    elapsed = time.perf_counter() - started

    winner_fitness = result.fitness[result.winner]
    nanometres = wavelengths_of(result.winner, wavelengths)
    write_stats(paths["stats"], result.generations, wavelengths)
    winner_path = winner_results_path(experiment, result.winner)
    write_winner(winner_path, result.winner, winner_fitness, identity, wavelengths)
    paths["summary"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": experiment,
                "primary_metric": "weighted_f1",
                "study_seed": config.STUDY_SEED,
                "winner": {
                    "selected_band_indices": list(result.winner),
                    "selected_wavelengths_nm": list(nanometres),
                    "candidate_seed": candidate_seed(result.winner, identity),
                    **{name: getattr(winner_fitness, name) for name in METRIC_FIELDS},
                },
                "elapsed_seconds": elapsed,
                "cache": {
                    "unique_candidates": len(result.fitness),
                    "hits": result.cache_hits,
                    "restored": cache.restored,
                    "evaluations": result.evaluations,
                },
                "libraries": library_versions(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    plot_fitness_evolution(paths["figure"], result.generations, nanometres)

    print(f"winner         {result.winner}")
    print("wavelengths_nm " + ", ".join(f"{value}" for value in nanometres))
    print(f"weighted_f1    {winner_fitness.weighted_f1!r}")
    print(
        f"unique {len(result.fitness)} | hits {result.cache_hits} | "
        f"restored {cache.restored} | evaluated {result.evaluations}"
    )
    print(f"elapsed_s      {elapsed:.1f}")
    for name in ("stats", "summary", "figure"):
        print(f"{name:<14} {paths[name]}")
    print(f"{'winner_csv':<14} {winner_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "experiment", nargs="?", type=int, help="experiment number of a search run"
    )
    parser.add_argument(
        "--evaluate",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        help="train and score one band triplet instead of searching",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.NUM_EPOCHS, help="epochs to train for (smoke use)"
    )
    parser.add_argument(
        "--population",
        type=int,
        default=config.POPULATION_SIZE,
        help="individuals per generation (smoke use)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=config.GENERATIONS,
        help="generations to evolve for (smoke use)",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="replay from the cache alone: any candidate that is not in it is an error",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="where to write the validation predictions; the default is under out/tables/"
        " and is never overwritten",
    )
    arguments = parser.parse_args(argv)

    if arguments.evaluate is not None and arguments.experiment is not None:
        parser.error("--evaluate takes no experiment number: it writes no exp_NN_ output")
    if arguments.evaluate is None and arguments.experiment is None:
        parser.error("give an experiment number to search, or --evaluate R G B to score one triplet")
    # A refused cache, a refused device or an occupied output path is a decision
    # this program made on purpose, so it reads as one line rather than as a
    # traceback the user has to interpret.
    try:
        if arguments.evaluate is not None:
            evaluate_command(
                tuple(arguments.evaluate),
                epochs=arguments.epochs,
                predictions_path=arguments.predictions,
            )
        else:
            search_command(
                arguments.experiment,
                population=arguments.population,
                generations=arguments.generations,
                epochs=arguments.epochs,
                evaluate=not arguments.no_evaluate,
            )
    except (ValueError, LookupError, FileExistsError) as refusal:
        print(f"ga.py: {refusal}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
