from __future__ import annotations

from typing import Any, Dict

import networkx as nx

from ..core_math import (
    entropy_degree,
    entropy_weights,
    entropy_confidence,
    network_entropy_rate,
    evolutionary_entropy_demetrius,
    fragility_from_entropy,
)


def calculate_all_entropies(G: nx.Graph) -> Dict[str, Any]:
    return {
        "H_deg": float(entropy_degree(G)),
        "H_w": float(entropy_weights(G)),
        "H_conf": float(entropy_confidence(G)),
        "H_rw": float(network_entropy_rate(G)),
        "H_evo": float(evolutionary_entropy_demetrius(G)),
    }


def fragility_from_H(H: float) -> float:
    return float(fragility_from_entropy(H))
