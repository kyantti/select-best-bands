"""Redraw the fitness curve of a finished search from the tables it left behind.

`ga.py` draws this figure at the end of a search.  This script draws the same
one again, from `exp_NN_ga_stats.csv` and `exp_NN_ga_summary.json` alone: no
GPU, no crops, no re-run.  It is how a figure is recovered after it is lost or
after the plotting code changes.

Usage:
    uv run python plot_fitness_evolution.py 21
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import ga

# The split of `ga.STATS_FIELDS` into how each column is read back.  Derived
# from that tuple, not typed out again, so a column added to the search cannot
# be dropped silently here.
COUNT_FIELDS = ga.STATS_FIELDS[:4]
SCORE_FIELDS = ga.STATS_FIELDS[4:8]
JSON_FIELDS = ga.STATS_FIELDS[8:]


def read_generations(path: Path) -> list[dict]:
    """The rows of one `exp_NN_ga_stats.csv`, typed as the plot wants them.

    Raises:
        ValueError: If the file is missing, empty, or has the columns of a
            search this code did not write.
    """
    if not path.exists():
        raise ValueError(f"{path} does not exist: run the search of that experiment first")
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        header = tuple(reader.fieldnames or ())
        if header != ga.STATS_FIELDS:
            raise ValueError(
                f"{path} has the columns {', '.join(header)}, which are not the ones "
                f"this search writes ({', '.join(ga.STATS_FIELDS)}). Experiments 1–20 "
                "belong to the earlier protocol; the curve of this one starts at 21."
            )
        rows = [
            {
                **{name: int(row[name]) for name in COUNT_FIELDS},
                **{name: float(row[name]) for name in SCORE_FIELDS},
                **{name: tuple(json.loads(row[name])) for name in JSON_FIELDS},
            }
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path} has a header and no generations")
    return rows


def winner_wavelengths(path: Path) -> tuple[float, ...]:
    """The winner's nanometres, which title the curve, from `exp_NN_ga_summary.json`.

    Raises:
        ValueError: If the summary is missing or names no winner.
    """
    if not path.exists():
        raise ValueError(f"{path} does not exist: the search of that experiment did not finish")
    summary = json.loads(path.read_text())
    nanometres = summary.get("winner", {}).get("selected_wavelengths_nm")
    if not nanometres:
        raise ValueError(f"{path} has no winner with selected_wavelengths_nm")
    return tuple(float(value) for value in nanometres)


def plot_command(experiment: int, *, verbose: bool = True) -> Path:
    """Redraw `exp_NN_fitness_evolution.png` from the tables of experiment `NN`."""
    paths = ga.experiment_paths(experiment)
    generations = read_generations(paths["stats"])
    nanometres = winner_wavelengths(paths["summary"])
    ga.plot_fitness_evolution(paths["figure"], generations, nanometres)
    if verbose:
        print(f"{len(generations)} generations from {paths['stats']}")
        print(f"wrote {paths['figure']}")
    return paths["figure"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", type=int, help="the experiment whose curve to redraw")
    arguments = parser.parse_args(argv)
    try:
        plot_command(arguments.experiment)
    except ValueError as refusal:
        print(f"plot_fitness_evolution.py: {refusal}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
