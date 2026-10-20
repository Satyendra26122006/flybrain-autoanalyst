"""Compare rule-based, real connectome, and shuffled-connectome routers.

Usage: python experiment.py [path/to/tensors.pt] [--datasets 300] [--epochs 15]
"""
import argparse

import numpy as np
import torch

from src import brain_router
from src.connectome import random_connectome, shuffled
from src.insights import TASKS
from src.profiling import profile
from src.router_rules import select_analyses as rules
from src.synthetic import build_training_set, label_dataset, make_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensors", nargs="?", help="Built FlyWire graph tensors; omit for a small synthetic graph")
    parser.add_argument("--datasets", type=int, default=300, help="Synthetic training datasets")
    parser.add_argument("--test-datasets", type=int, default=100, help="Held-out synthetic test datasets")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per training stage")
    parser.add_argument("--steps", type=int, default=5, help="Graph propagation steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Examples per optimizer step")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    args = parser.parse_args()

    graph = (
        torch.load(args.tensors, map_location="cpu", weights_only=True)
        if args.tensors
        else random_connectome()
    )
    x_train, y_train = build_training_set(args.datasets, seed=0)
    rng = np.random.default_rng(1)
    x_test, y_test, rule_predictions = [], [], []
    for _ in range(args.test_datasets):
        dataset = make_dataset(rng)
        features, labels = label_dataset(dataset)
        x_test.append(features)
        y_test.append(labels)
        dataset_profile = profile(dataset)
        selected = rules(dataset_profile)
        rule_predictions.append([float(task in selected) for task in TASKS])
    x_test = np.stack(x_test)
    y_test = np.stack(y_test)

    threshold_grid = (0.1, 0.2, 0.3, 0.4, 0.5)
    results = {
        "rule_based": {
            str(threshold): brain_router.evaluate(
                np.asarray(rule_predictions), y_test, threshold=threshold
            )
            for threshold in threshold_grid
        }
    }
    for name, candidate in (
        ("connectome", graph),
        ("shuffled", shuffled(graph)),
    ):
        model = brain_router.train(
            candidate,
            x_train,
            y_train,
            epochs=args.epochs,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
        )
        probabilities = brain_router.predict_probs(model, x_test)
        results[name] = {
            "probability_range": [
                float(probabilities.min()),
                float(probabilities.max()),
            ],
            "thresholds": {
                str(threshold): brain_router.evaluate(
                    probabilities, y_test, threshold=threshold
                )
                for threshold in threshold_grid
            },
        }
    print(results)


if __name__ == "__main__":
    main()
