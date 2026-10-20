"""Build and train the connectome analysis router from FlyWire connections."""
import argparse
from pathlib import Path

import torch

from src import brain_router
from src.connectome import build_tensors
from src.synthetic import build_training_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connections", default="data/raw/connections.csv.gz",
                        help="FlyWire Codex connections CSV or CSV.GZ")
    parser.add_argument("--tensors", default="data/processed/tensors.pt",
                        help="Output tensor path (or existing tensor input)")
    parser.add_argument("--checkpoint", default="checkpoints/brain_router.pt",
                        help="Trained router checkpoint output")
    parser.add_argument("--min-syn", type=int, default=1,
                        help="Minimum synapses per edge; use 1 to retain the full graph")
    parser.add_argument("--datasets", type=int, default=300,
                        help="Number of synthetic training datasets")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Training epochs per stage")
    parser.add_argument("--steps", type=int, default=5, help="Propagation steps per dataset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Training device (default: CUDA when available, else CPU)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tensor_path = Path(args.tensors)
    if tensor_path.is_file():
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    else:
        connections_path = Path(args.connections)
        if not connections_path.is_file():
            parser.error(
                f"Connections file not found: {connections_path}. "
                "Download the FlyWire Codex connections CSV/CSV.GZ and place it there."
            )
        tensors = build_tensors(connections_path, min_syn=args.min_syn, out_path=tensor_path)

    x_train, y_train = build_training_set(args.datasets, seed=args.seed)
    model = brain_router.train(
        tensors,
        x_train,
        y_train,
        epochs=args.epochs,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
    )
    brain_router.save_checkpoint(
        model,
        args.checkpoint,
        root_ids=tensors.get("root_ids"),
        metadata={
            "neurons": tensors["n"],
            "synapses": len(tensors["src"]),
            "training_datasets": args.datasets,
            "epochs_per_stage": args.epochs,
            "device": args.device,
        },
    )
    print(
        f"Saved {args.checkpoint}: {tensors['n']:,} neurons, "
        f"{len(tensors['src']):,} synapses."
    )


if __name__ == "__main__":
    main()
