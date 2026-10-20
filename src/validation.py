"""Every insight = claim, evidence, uncertainty, limitations, next step."""
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class Finding:
    tool: str
    claim: str
    evidence: dict
    uncertainty: str
    limitations: list = field(default_factory=list)
    next_step: str = ""
    p_adj: float | None = None

    def to_dict(self):
        return asdict(self)


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR adjustment."""
    p = np.asarray(pvals, float); n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in zip(range(n, 0, -1), order[::-1]):
        prev = min(prev, p[i] * n / rank)
        adj[i] = prev
    return adj


def corr_ci(r, n, z=1.96):
    if n < 4 or abs(r) >= 1:
        return (float(r), float(r))
    zr, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(zr - z * se)), float(np.tanh(zr + z * se))
