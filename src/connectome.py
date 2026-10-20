"""FlyWire connectome -> tensors, and a fully trainable connectome network.

Wiring is fixed. Every neuron (bias, gain) and every synapse (weight) is learnable.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

NEG_NT = {"GABA", "GLUT"}  # inhibitory (glutamate is inhibitory in fly CNS, approx.)


def _col(df, *cands):
    for c in df.columns:
        if any(k in c.lower() for k in cands):
            return c
    raise KeyError(f"none of {cands} in {list(df.columns)}")


def build_tensors(connections_path, min_syn=5, out_path=None):
    """Read Codex connections table -> src, dst, init_w, sign tensors."""
    df = pd.read_csv(connections_path)  # works with .csv or .csv.gz
    pre, post = _col(df, "pre_root", "pre_pt"), _col(df, "post_root", "post_pt")
    syn = _col(df, "syn_count")
    nt = next((c for c in df.columns if "nt_type" in c.lower()), None)
    df["_s"] = np.where(df[nt].isin(NEG_NT), -1.0, 1.0) if nt else 1.0
    g = df.groupby([pre, post]).agg(w=(syn, "sum"), s=("_s", "mean")).reset_index()
    g = g[g.w >= min_syn]
    ids = pd.unique(np.concatenate([g[pre].values, g[post].values]))
    index = {nid: i for i, nid in enumerate(ids)}
    t = {
        "src": torch.tensor(g[pre].map(index).values, dtype=torch.long),
        "dst": torch.tensor(g[post].map(index).values, dtype=torch.long),
        "init_w": torch.tensor(g.w.values, dtype=torch.float32),
        "sign": torch.tensor(np.where(g.s.values < 0, -1.0, 1.0), dtype=torch.float32),
        "n": len(ids), "root_ids": torch.tensor(ids.astype("int64")),
    }
    if out_path:
        torch.save(t, out_path)
    return t


def random_connectome(n=500, edges=5000, seed=0):
    """Small synthetic graph for tests/prototyping without downloaded data."""
    g = torch.Generator().manual_seed(seed)
    return {"src": torch.randint(0, n, (edges,), generator=g), "dst": torch.randint(0, n, (edges,), generator=g),
            "init_w": torch.randint(5, 50, (edges,), generator=g).float(),
            "sign": torch.where(torch.rand(edges, generator=g) < 0.7, 1.0, -1.0), "n": n}


def subset(t, keep):
    """Keep only neurons in `keep` (LongTensor of indices); re-index edges."""
    mask = torch.zeros(t["n"], dtype=torch.bool); mask[keep] = True
    e = mask[t["src"]] & mask[t["dst"]]
    remap = torch.full((t["n"],), -1, dtype=torch.long); remap[keep] = torch.arange(len(keep))
    return {"src": remap[t["src"][e]], "dst": remap[t["dst"][e]], "init_w": t["init_w"][e], "sign": t["sign"][e], "n": len(keep)}


def shuffled(t, seed=0):
    """Control: same neurons/edges/weights, destinations shuffled (breaks real wiring)."""
    g = torch.Generator().manual_seed(seed)
    return {**t, "dst": t["dst"][torch.randperm(len(t["dst"]), generator=g)]}


class Connectome(nn.Module):
    def __init__(self, t, n_in, n_out, input_idx=None, output_idx=None, steps=5, seed=0):
        super().__init__()
        n, g = t["n"], torch.Generator().manual_seed(seed)
        self.n, self.steps = n, steps
        self.register_buffer("src", t["src"]); self.register_buffer("dst", t["dst"])
        self.register_buffer("sign", t["sign"]); self.register_buffer("w0", t["init_w"].clone())
        in_deg = torch.zeros(n).index_add_(0, t["dst"], t["init_w"]).clamp(min=1)
        self.register_buffer("norm", 1.0 / in_deg[t["dst"]])           # per-target normalisation
        self.log_w = nn.Parameter(t["init_w"].log())                    # one weight per synapse
        self.bias = nn.Parameter(torch.zeros(n)); self.gain = nn.Parameter(torch.ones(n))  # per neuron
        self.input_idx = input_idx if input_idx is not None else torch.randperm(n, generator=g)[:min(n, 64 * n_in)]
        self.output_idx = output_idx if output_idx is not None else torch.randperm(n, generator=g)[:min(n, 32 * n_out)]
        self.encoder = nn.Linear(n_in, len(self.input_idx))
        self.readout = nn.Linear(len(self.output_idx), n_out)

    def forward(self, x):                      # x: (n_in,) -> (n_out,) logits
        inp = torch.zeros(self.n, device=x.device).index_add(0, self.input_idx.to(x.device), self.encoder(x))
        h = torch.zeros(self.n, device=x.device)
        w = self.log_w.exp() * self.sign * self.norm
        for _ in range(self.steps):
            agg = torch.zeros(self.n, device=x.device).index_add(0, self.dst, w * h[self.src])
            h = torch.tanh(self.gain * agg + inp + self.bias)
        return self.readout(h[self.output_idx.to(x.device)])

    def weight_penalty(self):                  # pull synapses toward real counts
        return ((self.log_w - self.w0.log()) ** 2).mean()

    def silence(self, neuron):                 # virtual neuron-removal experiment
        with torch.no_grad():
            self.gain[neuron] = 0.0; self.bias[neuron] = 0.0
