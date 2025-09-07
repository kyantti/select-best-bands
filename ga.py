"""
Genetic Algorithm for Optimal Band Selection in Hyperspectral Imaging

This script uses DEAP to evolve optimal RGB band combinations from hyperspectral data
for toxin classification using ResNet50 transfer learning.
"""

import array
import functools
import random
from collections.abc import Sequence
from itertools import repeat
import time
import numpy
from deap import algorithms, base, creator, tools
import torch
import sys
import torchvision
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pandas as pd

import cnn.data_setup
import cnn.engine
import cnn.util.helper_functions

# Experiment configuration
NUM_EXPERIMENTS = 5 
EXPERIMENT_START = 1 

# CNN Hyperparameters
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Hyperspectral data band range
START_BAND = 0
END_BAND = 447

# GA Hyperparameters
POPULATION_SIZE = 20
GENERATIONS = 50
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.15
ELITISM_SIZE = 1

# Setup directories containing the CSV files pointing to the hyperspectral data
train_csv = "train_dataset.csv"
test_csv = "test_dataset.csv"

# Cache for previously evaluated individuals to avoid redundant computations
history = {}


def evaluate(
    individual,
    device,
    train_transform,
    test_transform,
    train_data_samples,
    train_labels,
    test_data_samples,
    test_labels,
):
    """
    Generate RGB images from the individual and evaluate the model with the generated images
    :param individual: Individual to evaluate (3 band indices: [R, G, B])
    :param train_data_samples: Pre-loaded training hypercube data
    :param train_labels: Pre-loaded training labels
    :param test_data_samples: Pre-loaded test hypercube data
    :param test_labels: Pre-loaded test labels
    :return: Fitness of the individual (test accuracy)
    """
    # Check if the individual is in the history
    if tuple(individual) in history:
        print(f"Bands {list(individual)} found in history. Returning cached fitness.")
        cached_results = history[tuple(individual)]
        fitness = (
            cached_results["test_acc"][-1] if "test_acc" in cached_results else 0.0
        )
        return (fitness,)

    # Debug output to track progress
    r_band, g_band, b_band = int(individual[0]), int(individual[1]), int(individual[2])

    try:
        # Set band indices for this individual
        band_indices = [r_band, g_band, b_band]

        # Create DataLoaders with pre-loaded data
        train_dataloader = cnn.data_setup.create_train_dataloader(
            band_indices=band_indices,
            transform=train_transform,
            batch_size=BATCH_SIZE,
            num_workers=0,  # Ensure no subprocesses in Pool worker, otherwise may hang
            data_samples=train_data_samples,
            labels=train_labels,
            verbose=True,
        )

        test_dataloader = cnn.data_setup.create_test_dataloader(
            band_indices=band_indices,
            transform=test_transform,
            batch_size=BATCH_SIZE,
            num_workers=0,  # Ensure no subprocesses in Pool worker, otherwise may hang
            data_samples=test_data_samples,
            labels=test_labels,
            verbose=True,
        )

        # Model setup (fresh for each individual)
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights).to(device)

        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last two layers
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(
            in_features=in_features, out_features=4, bias=True
        ).to(device)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        start_time = timer()

        print(f"🔍 Starting evaluation of individual [{r_band}, {g_band}, {b_band}]")

        # Setup training and save the results
        results = cnn.engine.train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=NUM_EPOCHS,
            verbose=True,
            device=device,
        )

        end_time = timer()

        print(f"⏱️ Total training time: {end_time - start_time:.3f} seconds")

        # Extract fitness (e.g., best test accuracy)
        fitness = results["test_acc"][-1] if "test_acc" in results else 0.0

        print(f"📊 Bands [{r_band}, {g_band}, {b_band}] -> Fitness: {fitness:.4f}")

        # Store the result in the history
        history[tuple(individual)] = results

        # DEAP expects a tuple
        return (fitness,)

    except Exception as e:
        print(f"❌ Error evaluating individual {list(individual)}: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sys.exit(1)


def cx_blend_clamped(ind1, ind2, alpha, START_BAND, END_BAND):
    """Executes a blend crossover that modify in-place the input individuals.
    The blend crossover expects :term:`sequence` individuals of floating point
    numbers.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param alpha: Extent of the interval in which the new values can be drawn
                  for each attribute on both side of the parents' attributes.
    :returns: A tuple of two individuals.

    If an individual value is outside of the interval [0, 111], it is clipped to
    the nearest value inside the interval

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i, (x1, x2) in enumerate(zip(list(ind1), list(ind2))):
        gamma = (1.0 + 2.0 * alpha) * random.random() - alpha
        ind1[i] = int((1.0 - gamma) * x1 + gamma * x2)
        ind2[i] = int(gamma * x1 + (1.0 - gamma) * x2)

        # Clamp values to domain
        ind1[i] = min(max(ind1[i], START_BAND), END_BAND)
        ind2[i] = min(max(ind2[i], START_BAND), END_BAND)

    return ind1, ind2


def mut_gaussian_clamped(individual, mu, sigma, indpb, START_BAND, END_BAND):
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.

    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.

    If an individual value is outside of the interval [0, 111], it is clipped to
    the nearest value inside the interval

    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError(
            "mu must be at least the size of individual: %d < %d" % (len(mu), size)
        )
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError(
            "sigma must be at least the size of individual: %d < %d"
            % (len(sigma), size)
        )

    for i, m, s in zip(range(size), mu, sigma):
        if random.random() < indpb:
            individual[i] += int(random.gauss(m, s))
            # Clamp the mutated value within the domain
            individual[i] = min(max(individual[i], START_BAND), END_BAND)

    return (individual,)


def run_single_experiment(seed, experiment_num):
    print(f"🎯 Optimizing RGB band selection from {END_BAND + 1} total bands")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print("📊 Pre-loading hypercubes...")

    train_data_samples, train_labels = cnn.data_setup.load_hypercubes_from_csv(
        train_csv
    )

    test_data_samples, test_labels = cnn.data_setup.load_hypercubes_from_csv(test_csv)

    print("✅ All hypercubes loaded into memory!")

    # Setup first device available with enough memory
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

    random.seed(seed)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="h", fitness=creator.FitnessMax)  # type: ignore

    toolbox = base.Toolbox()

    # Tell DEAP to use our pool for parallel evaluation
    # pool = multiprocessing.Pool(processes=num_workers)
    # toolbox.register("map", pool.map)
    # num_workers = 1

    # Attribute generator
    toolbox.register("attr_int", random.randint, START_BAND, END_BAND)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 3)  # type: ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore

    # Pass pre-loaded data along with other parameters to evaluate
    toolbox.register(
        "evaluate",
        functools.partial(
            evaluate,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            train_data_samples=train_data_samples,
            train_labels=train_labels,
            test_data_samples=test_data_samples,
            test_labels=test_labels,
        ),
    )
    toolbox.register(
        "mate",
        lambda ind1, ind2: cx_blend_clamped(
            ind1, ind2, alpha=0.5, START_BAND=START_BAND, END_BAND=END_BAND
        ),
    )
    toolbox.register(
        "mutate",
        lambda ind: mut_gaussian_clamped(
            ind,
            mu=0,
            sigma=1.0,
            indpb=0.1,
            START_BAND=START_BAND,
            END_BAND=END_BAND,
        ),
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=POPULATION_SIZE)  # type: ignore

    halloffame = tools.HallOfFame(ELITISM_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    stats.register("best", lambda pop: halloffame[0])

    # Run the genetic algorithm
    start_time = time.time()

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=CROSSOVER_PROB,
        mutpb=MUTATION_PROB,
        ngen=GENERATIONS,
        stats=stats,
        halloffame=halloffame,
        verbose=True,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"⏱️ Total runtime: {elapsed_time / 60:.1f} minutes ({elapsed_time:.1f} seconds)")

    print(f"🏆 Best individual: {halloffame[0]} -> Fitness: {halloffame[0].fitness.values[0]}")

    # The 'best' column is now automatically recorded by the logbook
    df_stats = pd.DataFrame(logbook)
    # The 'best' column contains Individual objects, convert them to simple lists
    df_stats["best"] = df_stats["best"].apply(list)
    csv_path = f"out/tables/exp_{experiment_num:02d}_ga_stats.csv"
    df_stats.to_csv(csv_path, index=False)

    print(f"📊 GA statistics saved to '{csv_path}'")

    # Save plotting data and final results for the best individual
    best_bands = list(halloffame[0])

    if tuple(best_bands) in history:

        best_results = history[tuple(best_bands)]

        # Create a dictionary with the final metrics from the last epoch
        final_metrics = {
            "train_loss": best_results["train_loss"][-1],
            "train_acc": best_results["train_acc"][-1],
            "test_loss": best_results["test_loss"][-1],
            "test_acc": best_results["test_acc"][-1],
        }

        # Convert the dictionary to a pandas DataFrame
        df_final_results = pd.DataFrame([final_metrics])

        # Define the CSV filename
        csv_filename = f"out/tables/exp_{experiment_num:02d}_cnn_results_{best_bands[0]}_{best_bands[1]}_{best_bands[2]}.csv"

        # Save the DataFrame to a CSV file
        df_final_results.to_csv(csv_filename, index=False)

        print(f"📈 CNN training results for best individual saved to '{csv_filename}'")

        cnn.util.helper_functions.plot_loss_curves(best_results)
        plot_filename = f"out/figures/exp_{experiment_num:03d}_cnn_results_{best_bands[0]}_{best_bands[1]}_{best_bands[2]}.png"
        plt.savefig(
            plot_filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(f"📈 Loss curves plots for the best individual saved to '{plot_filename}'")


def main(seed):
    for i in range(NUM_EXPERIMENTS):
        experiment_num = EXPERIMENT_START + i

        print(f"🧬 EXPERIMENT {experiment_num:02d}")

        # Use different seeds for each experiment to ensure different results
        current_seed = seed + i  # Base seed + experiment offset
        run_single_experiment(current_seed, experiment_num)

        print(f"✅ Experiment {experiment_num:02d} completed!")


if __name__ == "__main__":
    main(64)
