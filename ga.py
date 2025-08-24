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

# Import our custom modules
from cnn.util.data_setup import load_hypercubes_to_memory
from cnn.transfer_learning import eval


def evaluate(individual):
    """
    Generate RGB images from the individual and evaluate the model with the generated images
    :param individual: Individual to evaluate (3 band indices: [R, G, B])
    :return: Fitness of the individual (test accuracy)
    """
    # Debug output to track progress
    r_band, g_band, b_band = int(individual[0]), int(individual[1]), int(individual[2])
    print(f"🔍 Evaluating bands [{r_band}, {g_band}, {b_band}]...")

    try:
        # Evaluate using our memory-based ResNet50 training with GPU optimization
        # The fitness function now uses the global in-memory hypercubes
        fitness = eval(
            r_band, g_band, b_band, epochs=30, batch_size=256, learning_rate=0.001, verbose=False
        )

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


def main(seed=42, domain_min=0, domain_max=63):
    """
    Main function to run the genetic algorithm
    :param seed: Seed for the random number generator
    :param domain_min: Minimum band index (default: 0)
    :param domain_max: Maximum band index (default: 447 for 448 bands)
    """

    print(f"🎯 Optimizing RGB band selection from {domain_max + 1} total bands")
    print(f"🔢 Band range: {domain_min} to {domain_max}")

    # Load data once before starting GA
    load_hypercubes_to_memory(
        train_dir="data/processed/train", test_dir="data/processed/test"
    )

    random.seed(seed)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="h", fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_int", random.randint, domain_min, domain_max)

    # Structure initializers
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 3
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", functools.partial(evaluate))
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

    population_size = 25
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
        verbose=True,  # Enable verbose output
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(
        f"⏱️  Total runtime: {elapsed_time / 60:.1f} minutes ({elapsed_time:.1f} seconds)"
    )

    return hof[0], hof[0].fitness.values[0]


if __name__ == "__main__":
    best_individual, best_fitness = main()
