"""
Genetic Algorithm for Optimal Band Selection in Hyperspectral Imaging

This script uses DEAP to evolve optimal RGB band combinations from hyperspectral data
for toxin classification using ResNet50 transfer learning.
"""

import array
import functools
import multiprocessing
import random
from collections.abc import Sequence
from itertools import repeat
import time
import numpy
from deap import algorithms, base, creator, tools
import torch
import sys
import torchvision

import cnn.data_setup
import cnn.engine


# Define constants (ensure these are defined elsewhere or set here)
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20  # For GA, use 1 epoch for speed

# Setup directories
train_csv = "train_dataset.csv"
test_csv = "test_dataset.csv"


def evaluate(
    individual, model, optimizer, loss_fn, device, train_transform, test_transform
):
    """
    Generate RGB images from the individual and evaluate the model with the generated images
    :param individual: Individual to evaluate (3 band indices: [R, G, B])
    :return: Fitness of the individual (test accuracy)
    """
    # Debug output to track progress
    r_band, g_band, b_band = int(individual[0]), int(individual[1]), int(individual[2])
    print(f"🔍 Evaluating bands [{r_band}, {g_band}, {b_band}]...")

    try:
        # Set band indices for this individual
        band_indices = [r_band, g_band, b_band]

        # Create DataLoaders with help from data_setup.py
        train_dataloader = cnn.data_setup.create_train_dataloader(
            train_csv=train_csv,
            band_indices=band_indices,
            transform=train_transform,
            batch_size=BATCH_SIZE,
            num_workers=0,  # Ensure no subprocesses in Pool worker
        )

        test_dataloader = cnn.data_setup.create_test_dataloader(
            test_csv=test_csv,
            band_indices=band_indices,
            transform=test_transform,
            batch_size=BATCH_SIZE,
            num_workers=0,  # Ensure no subprocesses in Pool worker
        )

        # Setup training and save the results
        results = cnn.engine.train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=NUM_EPOCHS,
            device=device,
        )

        # Extract fitness (e.g., best test accuracy)
        fitness = results["test_acc"][-1] if "test_acc" in results else 0.0

        print(f"✅ Bands [{r_band}, {g_band}, {b_band}] -> Fitness: {fitness:.4f}")

        # DEAP expects a tuple
        return (fitness,)

    except Exception as e:
        print(f"❌ Error evaluating individual {list(individual)}: {e}")
        # Clear GPU cache on error too
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # End the program
        sys.exit(1)


def cx_blend_clamped(ind1, ind2, alpha, domain_min, domain_max):
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
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        gamma = (1.0 + 2.0 * alpha) * random.random() - alpha
        ind1[i] = int((1.0 - gamma) * x1 + gamma * x2)
        ind2[i] = int(gamma * x1 + (1.0 - gamma) * x2)

        # Clamp values to domain
        ind1[i] = min(max(ind1[i], domain_min), domain_max)
        ind2[i] = min(max(ind2[i], domain_min), domain_max)

    return ind1, ind2


def mut_gaussian_clamped(individual, mu, sigma, indpb, domain_min, domain_max):
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
            individual[i] = min(max(individual[i], domain_min), domain_max)

    return (individual,)


def main(seed=123, domain_min=0, domain_max=447):
    """
    Main function to run the genetic algorithm
    :param seed: Seed for the random number generator
    :param domain_min: Minimum band index (default: 0)
    :param domain_max: Maximum band index (default: 447 for 448 bands)
    """

    print(f"🎯 Optimizing RGB band selection from {domain_max + 1} total bands")
    print(f"🔢 Band range: {domain_min} to {domain_max}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Setup target device and model inside main()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Model setup
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

    # Get the number of features from the last layer
    in_features = model.fc.in_features

    # Replace the final classifier with a simple linear layer for 4 classes
    model.fc = torch.nn.Linear(in_features=in_features, out_features=4, bias=True).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create a pool of workers (adjust number based on GPU capacity)
    num_workers = 12

    random.seed(seed)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="h", fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Tell DEAP to use our pool for parallel evaluation
    pool = multiprocessing.Pool(processes=num_workers)
    toolbox.register("map", pool.map)

    # Attribute generator
    toolbox.register("attr_int", random.randint, domain_min, domain_max)

    # Structure initializers
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 3
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Pass model, optimizer, loss_fn, device, transforms to evaluate
    toolbox.register(
        "evaluate",
        functools.partial(
            evaluate,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
        ),
    )
    toolbox.register(
        "mate",
        lambda ind1, ind2: cx_blend_clamped(
            ind1, ind2, alpha=0.5, domain_min=domain_min, domain_max=domain_max
        ),
    )
    toolbox.register(
        "mutate",
        lambda ind: mut_gaussian_clamped(
            ind,
            mu=0,
            sigma=1.0,
            indpb=0.1,
            domain_min=domain_min,
            domain_max=domain_max,
        ),
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    population_size = 24
    generations = 50

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Run the genetic algorithm
    start_time = time.time()

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.8,
        mutpb=0.15,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(
        f"⏱️  Total runtime: {elapsed_time / 60:.1f} minutes ({elapsed_time:.1f} seconds)"
    )

    return hof[0], hof[0].fitness.values[0]


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    best_individual, best_fitness = main()
