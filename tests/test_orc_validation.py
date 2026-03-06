"""Валидация ORC-реализации на каноничных графах."""

import networkx as nx
import numpy as np
import pytest

from src.core_math import ollivier_ricci_edge, ollivier_ricci_summary


class TestORCCanonical:
    """Проверяем теоретически ожидаемые свойства ORC."""

    def test_complete_graph_positive(self):
        """В полном графе средняя кривизна должна быть положительной."""
        graph = nx.complete_graph(8)
        for u, v in graph.edges():
            graph[u][v]["weight"] = 1.0

        curv = ollivier_ricci_summary(graph, sample_edges=100, force_sequential=True)
        assert curv.kappa_mean > 0, f"Expected positive κ in K_8, got {curv.kappa_mean}"
        assert curv.kappa_frac_negative == 0.0

    def test_star_graph_non_positive(self):
        """Звезда (как дерево) не должна давать устойчиво положительную κ."""
        graph = nx.star_graph(10)
        for u, v in graph.edges():
            graph[u][v]["weight"] = 1.0

        curv = ollivier_ricci_summary(graph, sample_edges=100, force_sequential=True)
        assert curv.kappa_mean <= 0, f"Expected non-positive κ in star, got {curv.kappa_mean}"

    def test_cycle_near_zero(self):
        """Цикл должен иметь κ около нуля."""
        graph = nx.cycle_graph(20)
        for u, v in graph.edges():
            graph[u][v]["weight"] = 1.0

        curv = ollivier_ricci_summary(graph, sample_edges=100, force_sequential=True)
        assert abs(curv.kappa_mean) < 0.2, f"Expected near-zero κ in cycle, got {curv.kappa_mean}"

    def test_random_vs_structured(self):
        """Barbell должен иметь большую долю отрицательной κ, чем random с тем же E."""
        graph_barbell = nx.barbell_graph(8, 1)
        for u, v in graph_barbell.edges():
            graph_barbell[u][v]["weight"] = 1.0

        graph_random = nx.gnm_random_graph(17, graph_barbell.number_of_edges(), seed=42)
        for u, v in graph_random.edges():
            graph_random[u][v]["weight"] = 1.0

        k_barbell = ollivier_ricci_summary(graph_barbell, sample_edges=100, force_sequential=True)
        k_random = ollivier_ricci_summary(graph_random, sample_edges=100, force_sequential=True)
        assert k_barbell.kappa_frac_negative > k_random.kappa_frac_negative

    def test_extended_fields_present(self):
        """Расширенные поля CurvatureSummary должны быть заполнены."""
        graph = nx.karate_club_graph()
        for u, v in graph.edges():
            graph[u][v]["weight"] = 1.0

        curv = ollivier_ricci_summary(graph, sample_edges=50, force_sequential=True)
        assert hasattr(curv, "kappa_var")
        assert hasattr(curv, "kappa_skew")
        assert hasattr(curv, "kappa_entropy")
        assert hasattr(curv, "kappa_values")
        assert len(curv.kappa_values) > 0


class TestORCAgainstReference:
    """Сравнение с GraphRicciCurvature (если пакет установлен)."""

    @pytest.fixture
    def grc_available(self):
        try:
            from GraphRicciCurvature.OllivierRicci import OllivierRicci  # noqa: F401
            return True
        except ImportError:
            pytest.skip("GraphRicciCurvature not installed")

    def test_karate_correlation(self, grc_available):
        """Корреляция с reference-реализацией должна быть высокой."""
        from GraphRicciCurvature.OllivierRicci import OllivierRicci

        graph = nx.karate_club_graph()
        for u, v in graph.edges():
            graph[u][v]["weight"] = 1.0

        # Reference implementation.
        ref_orc = OllivierRicci(graph.copy(), alpha=0.0)
        ref_orc.compute_ricci_curvature()
        ref_vals = {(u, v): ref_orc.G[u][v].get("ricciCurvature", 0.0) for u, v in graph.edges()}

        # Current implementation.
        our_vals = {}
        for u, v in graph.edges():
            kappa = ollivier_ricci_edge(graph, u, v)
            if kappa is not None:
                our_vals[(u, v)] = kappa

        common = set(ref_vals.keys()) & set(our_vals.keys())
        assert len(common) >= 20, f"Too few common edges: {len(common)}"

        ref_arr = np.array([ref_vals[e] for e in common])
        our_arr = np.array([our_vals[e] for e in common])
        corr = np.corrcoef(ref_arr, our_arr)[0, 1]
        assert corr > 0.8, f"Correlation with reference too low: {corr:.3f}"
