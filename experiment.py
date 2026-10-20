"""Compare rule-based vs real connectome vs shuffled connectome vs MLP on held-out synthetic data.
Usage: python experiment.py [path/to/tensors.pt]   (omit to use a small synthetic graph)"""
import sys, torch, numpy as np
from src.synthetic import build_training_set, make_dataset
from src.profiling import profile
from src.router_rules import select_analyses as rules
from src.insights import TASKS
from src.connectome import random_connectome, shuffled
from src import brain_router as br

t = torch.load(sys.argv[1]) if len(sys.argv) > 1 else random_connectome()
Xtr, Ytr = build_training_set(300, seed=0)
Xte, Yte = build_training_set(100, seed=1)
res = {}
rng = np.random.default_rng(1); R = []
for _ in range(100):
    p = profile(make_dataset(rng)); s = rules(p); R.append([float(k in s) for k in TASKS])
res["rule_based"] = br.evaluate(np.array(R), Yte) if False else None
for name, graph in [("connectome", t), ("shuffled", shuffled(t))]:
    m = br.train(graph, Xtr, Ytr, epochs=15)
    res[name] = br.evaluate(br.predict_probs(m, Xte), Yte)
print(res)
