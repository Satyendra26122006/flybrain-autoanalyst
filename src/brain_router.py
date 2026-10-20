"""Version C: connectome network chooses which analyses to run."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from .connectome import Connectome
from .profiling import feature_vector, N_FEATURES
from .insights import TASKS


CHECKPOINT_VERSION = 1


def train(
    t,
    X,
    Y,
    epochs=30,
    lr=3e-3,
    l2=1.0,
    steps=5,
    device="cpu",
    staged=True,
    seed=0,
    batch_size=16,
):
    if batch_size < 1:
        raise ValueError("Training batch size must be positive.")
    torch.manual_seed(seed)
    m = Connectome(t, N_FEATURES, len(TASKS), steps=steps, seed=seed).to(device)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    Y = torch.as_tensor(Y, dtype=torch.float32, device=device)
    for stage in (["io", "all"] if staged else ["all"]):   # io first, then every neuron/synapse
        for n_, p in m.named_parameters():
            p.requires_grad = (stage == "all") or n_.startswith(("encoder", "readout"))
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
        for ep in range(epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                batch = perm[i:i + batch_size]
                opt.zero_grad(set_to_none=True)
                for index in batch:
                    loss = (
                        F.binary_cross_entropy_with_logits(m(X[index]), Y[index])
                        + l2 * 1e-2 * m.weight_penalty()
                    )
                    (loss / len(batch)).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
    return m


def save_checkpoint(m, path, root_ids=None, metadata=None):
    """Save a portable state-dict checkpoint for app inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "n_features": N_FEATURES,
        "tasks": TASKS,
        "steps": m.steps,
        "state_dict": m.state_dict(),
        "input_idx": m.input_idx.detach().cpu(),
        "output_idx": m.output_idx.detach().cpu(),
        "metadata": metadata or {},
    }
    if root_ids is not None:
        payload["root_ids"] = root_ids.detach().cpu()
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    """Load a connectome router checkpoint without unpickling Python objects."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload["format_version"] != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported brain checkpoint version: {payload['format_version']}")
    if payload["n_features"] != N_FEATURES or payload["tasks"] != TASKS:
        raise ValueError("Brain checkpoint does not match the current feature or task definitions.")

    state = payload["state_dict"]
    graph = {
        "src": state["src"],
        "dst": state["dst"],
        "init_w": state["w0"],
        "sign": state["sign"],
        "n": len(state["bias"]),
    }
    model = Connectome(
        graph,
        N_FEATURES,
        len(TASKS),
        input_idx=payload["input_idx"],
        output_idx=payload["output_idx"],
        steps=payload["steps"],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_probs(m, X):
    device = next(m.parameters()).device
    return torch.stack([torch.sigmoid(m(torch.as_tensor(x, device=device))) for x in X]).cpu().numpy()


def select_analyses(m, p, threshold=0.5):
    probs = predict_probs(m, [feature_vector(p)])[0]
    return [t for t, pr in zip(TASKS, probs) if pr >= threshold]


def evaluate(probs, Y, threshold=0.5):
    pred = probs >= threshold
    tp = (pred & (Y > .5)).sum(); fp = (pred & (Y <= .5)).sum(); fn = (~pred & (Y > .5)).sum()
    return {"precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)), "tools_per_dataset": float(pred.sum(1).mean())}
