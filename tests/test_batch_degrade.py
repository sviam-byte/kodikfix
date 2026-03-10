"""Tests for batch_degrade module (degradation + optional phenotype pass)."""

from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx
import pandas as pd

from src.batch_degrade import (
    ALL_ATTACK_KINDS,
    _already_done,
    _item_csv_name,
    consolidate,
    run_batch_degrade,
    run_phenotype_pass,
)


def _toy_graph() -> nx.Graph:
    """Create a small weighted graph with two dense modules and weak bridges."""
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=3.0)
    graph.add_edge(1, 2, weight=3.0)
    graph.add_edge(0, 2, weight=3.0)
    graph.add_edge(3, 4, weight=3.0)
    graph.add_edge(4, 5, weight=3.0)
    graph.add_edge(3, 5, weight=3.0)
    graph.add_edge(2, 3, weight=0.2)
    graph.add_edge(1, 4, weight=0.2)
    return graph


def _fake_sz_df() -> pd.DataFrame:
    """Build a deterministic SZ reference table for phenotype tests."""
    return pd.DataFrame(
        [
            {"density": 0.3, "clustering": 0.4, "mod": 0.6},
            {"density": 0.25, "clustering": 0.35, "mod": 0.55},
        ]
    )


def _fake_hc_df() -> pd.DataFrame:
    """Build a deterministic HC baseline table for scale normalization tests."""
    return pd.DataFrame(
        [
            {"density": 0.5, "clustering": 0.55, "mod": 0.48},
            {"density": 0.53, "clustering": 0.56, "mod": 0.47},
        ]
    )


def test_item_csv_name() -> None:
    """Filename helper should preserve attack and sanitize subject id."""
    assert _item_csv_name("subj_001", "random_edges") == "subj_001__random_edges.csv"
    assert "/" not in _item_csv_name("path/to/file", "attack")


def test_run_batch_degrade_basic_backward_compat() -> None:
    """Default return value should remain Path for backward compatibility."""
    with TemporaryDirectory() as temp_dir:
        output = run_batch_degrade(
            [_toy_graph(), _toy_graph()],
            subject_ids=["s1", "s2"],
            attack_kinds=["random_edges", "inter_module_removal"],
            out_dir=temp_dir,
            steps=3,
            frac=0.5,
            seed=42,
            eff_sources_k=4,
            compute_heavy_every=1,
            metric_names=["density", "mod"],
        )
        assert isinstance(output, Path)
        df = pd.read_csv(output)

    assert len(df) == 2 * 2 * 4
    assert set(df.attack_kind.unique()) == {"random_edges", "inter_module_removal"}


def test_run_with_phenotype_pass() -> None:
    """Batch run with SZ/HC inputs should execute phenotype pass and return details."""
    with TemporaryDirectory() as temp_dir:
        result = run_batch_degrade(
            [_toy_graph(), _toy_graph()],
            subject_ids=["s1", "s2"],
            attack_kinds=["random_edges", "weight_noise"],
            out_dir=temp_dir,
            steps=3,
            frac=0.5,
            seed=42,
            eff_sources_k=4,
            compute_heavy_every=1,
            metric_names=["density", "clustering", "mod"],
            sz_group_metrics_df=_fake_sz_df(),
            hc_baseline_metrics_df=_fake_hc_df(),
            phenotype_metrics=["density", "clustering", "mod"],
            return_details=True,
        )

    assert result["phenotype"] is not None
    phenotype = result["phenotype"]
    assert "target_vector" in phenotype
    assert "summary_attack" in phenotype
    assert not phenotype["summary_attack"].empty
    assert not phenotype["scalar_summary"].empty
    assert "distance_to_target" in phenotype["trajectory_results"].columns


def test_phenotype_pass_standalone() -> None:
    """Standalone phenotype pass should work on previously saved trajectories."""
    with TemporaryDirectory() as temp_dir:
        trajectories_csv = run_batch_degrade(
            [_toy_graph()],
            subject_ids=["s1"],
            attack_kinds=["random_edges"],
            out_dir=temp_dir,
            steps=3,
            frac=0.5,
            seed=42,
            eff_sources_k=4,
            compute_heavy_every=1,
            metric_names=["density", "clustering", "mod"],
        )
        phenotype = run_phenotype_pass(
            trajectories_csv=trajectories_csv,
            sz_group_metrics_df=_fake_sz_df(),
            hc_baseline_metrics_df=_fake_hc_df(),
            phenotype_metrics=["density", "clustering", "mod"],
        )

    assert "target_vector" in phenotype
    assert not phenotype["subject_results"].empty


def test_skip_existing() -> None:
    """Second run with skip_existing should reuse existing per-item output."""
    with TemporaryDirectory() as temp_dir:
        run_batch_degrade(
            [_toy_graph()],
            subject_ids=["s1"],
            attack_kinds=["random_edges"],
            out_dir=temp_dir,
            steps=3,
            frac=0.5,
            seed=42,
            eff_sources_k=4,
            compute_heavy_every=1,
            metric_names=["density"],
        )
        assert _already_done(Path(temp_dir) / "per_item", "s1", "random_edges")

        result = run_batch_degrade(
            [_toy_graph()],
            subject_ids=["s1"],
            attack_kinds=["random_edges"],
            out_dir=temp_dir,
            steps=3,
            frac=0.5,
            seed=42,
            eff_sources_k=4,
            compute_heavy_every=1,
            metric_names=["density"],
            skip_existing=True,
            return_details=True,
        )

    assert result["n_skipped"] == 1


def test_start_from() -> None:
    """start_from should skip preceding subjects completely."""
    with TemporaryDirectory() as temp_dir:
        trajectories_csv = run_batch_degrade(
            [_toy_graph(), _toy_graph(), _toy_graph()],
            subject_ids=["s0", "s1", "s2"],
            attack_kinds=["random_edges"],
            out_dir=temp_dir,
            steps=3,
            frac=0.5,
            seed=42,
            eff_sources_k=4,
            compute_heavy_every=1,
            metric_names=["density"],
            start_from=2,
        )
        df = pd.read_csv(trajectories_csv)

    assert set(df.subject_id.unique()) == {"s2"}


def test_consolidate_standalone() -> None:
    """Consolidate helper should rebuild trajectories_all.csv from per-item outputs."""
    with TemporaryDirectory() as temp_dir:
        run_batch_degrade(
            [_toy_graph()],
            subject_ids=["s1"],
            attack_kinds=["random_edges"],
            out_dir=temp_dir,
            steps=3,
            frac=0.5,
            seed=42,
            eff_sources_k=4,
            compute_heavy_every=1,
            metric_names=["density"],
        )
        out_path = consolidate(temp_dir)
        df = pd.read_csv(out_path)

    assert len(df) == 4


def test_all_attack_kinds_run() -> None:
    """Smoke test: every declared attack kind should complete on toy graph."""
    with TemporaryDirectory() as temp_dir:
        result = run_batch_degrade(
            [_toy_graph()],
            subject_ids=["smoke"],
            attack_kinds=ALL_ATTACK_KINDS,
            out_dir=temp_dir,
            steps=2,
            frac=0.3,
            seed=42,
            eff_sources_k=4,
            compute_heavy_every=1,
            metric_names=["density", "clustering", "mod"],
            return_details=True,
        )
        df = pd.read_csv(result["trajectories_csv"])

    assert set(df.attack_kind.unique()) == set(ALL_ATTACK_KINDS)
    assert len(result["errors"]) == 0
