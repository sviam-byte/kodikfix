from __future__ import annotations

import networkx as nx

from ..attacks import run_attack, run_edge_attack
from ..attacks_mix import run_mix_attack


class AttackService:
    @staticmethod
    def run_node_attack(G: nx.Graph, *args, **kwargs):
        return run_attack(G, *args, **kwargs)

    @staticmethod
    def run_edge_attack(G: nx.Graph, *args, **kwargs):
        return run_edge_attack(G, *args, **kwargs)

    @staticmethod
    def run_mix_attack(G: nx.Graph, *args, **kwargs):
        return run_mix_attack(G, *args, **kwargs)
