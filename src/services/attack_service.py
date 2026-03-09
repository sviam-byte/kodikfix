from __future__ import annotations

import networkx as nx

from ..attacks import run_attack, run_edge_attack
from ..attacks_mix import run_mix_attack
from ..degradation import prepare_module_info, run_degradation_trajectory
from ..phenotype_matching import compare_degradation_models


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

    @staticmethod
    def prepare_module_info(G: nx.Graph, *args, **kwargs):
        return prepare_module_info(G, *args, **kwargs)

    @staticmethod
    def run_degradation_trajectory(G: nx.Graph, *args, **kwargs):
        return run_degradation_trajectory(G, *args, **kwargs)

    @staticmethod
    def compare_degradation_models(hc_graphs, *args, **kwargs):
        return compare_degradation_models(hc_graphs, *args, **kwargs)
