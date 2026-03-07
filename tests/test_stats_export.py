import io
from zipfile import ZipFile

import pandas as pd

from src.state_models import build_experiment_entry, build_graph_entry
from src.stats_export import (
    build_mixfrac_subjects_table,
    build_subject_metrics_table,
    build_trajectories_long_table,
    export_stats_xlsx_bytes,
    export_stats_zip_bytes,
)


def _edges_df() -> pd.DataFrame:
    """Build a tiny connected graph for export smoke tests."""
    return pd.DataFrame(
        {
            "src": ["a", "b", "c", "d", "a"],
            "dst": ["b", "c", "d", "a", "c"],
            "weight": [1, 1, 1, 1, 1],
            "confidence": [1, 1, 1, 1, 1],
        }
    )


def test_stats_tables_shapes_and_columns():
    """Ensure core table builders return expected basic structure."""
    g_hc = build_graph_entry(
        name="hc_001",
        source="upload",
        edges=_edges_df(),
        src_col="src",
        dst_col="dst",
        entry_id="g_hc",
    )
    g_sz = build_graph_entry(
        name="sz_001",
        source="upload",
        edges=_edges_df(),
        src_col="src",
        dst_col="dst",
        entry_id="g_sz",
    )
    graphs = {g_hc.id: g_hc, g_sz.id: g_sz}

    mixfrac_exp = build_experiment_entry(
        name="mixfrac",
        graph_id=g_sz.id,
        attack_kind="mix_frac_estimate",
        params={
            "attack_family": "mixfrac",
            "mix_frac_star": 0.4,
            "used_metrics": ["clustering"],
        },
        history=pd.DataFrame({"mix_frac_star": [0.4]}),
        entry_id="e_mix",
    )
    traj_exp = build_experiment_entry(
        name="traj",
        graph_id=g_sz.id,
        attack_kind="hrish_mix",
        params={"attack_family": "mix"},
        history=pd.DataFrame(
            {
                "step": [0, 1],
                "mix_frac_effective": [0.0, 0.1],
                "clustering": [0.2, 0.18],
            }
        ),
        entry_id="e_traj",
    )
    experiments = [mixfrac_exp, traj_exp]

    df_subjects = build_subject_metrics_table(
        graphs,
        min_conf=0,
        min_weight=0,
        analysis_mode="Global",
        compute_curvature=False,
    )
    assert len(df_subjects) == 2
    assert "subject_id" in df_subjects.columns

    df_mix = build_mixfrac_subjects_table(experiments, graphs)
    assert len(df_mix) == 1
    assert float(df_mix.loc[0, "mix_frac_star"]) == 0.4

    df_traj = build_trajectories_long_table(experiments, graphs)
    assert not df_traj.empty
    assert set(["metric", "value", "x_kind", "x_value"]).issubset(df_traj.columns)


def test_stats_zip_and_xlsx_exports():
    """Ensure both export formats include full table bundle."""
    g = build_graph_entry(
        name="hc_001",
        source="upload",
        edges=_edges_df(),
        src_col="src",
        dst_col="dst",
        entry_id="g1",
    )
    e = build_experiment_entry(
        name="traj",
        graph_id="g1",
        attack_kind="degree",
        params={"attack_family": "node"},
        history=pd.DataFrame({"step": [0, 1], "removed_frac": [0.0, 0.1], "lcc_frac": [1.0, 0.9]}),
        entry_id="e1",
    )

    blob_zip = export_stats_zip_bytes(
        {"g1": g},
        [e],
        min_conf=0,
        min_weight=0,
        analysis_mode="Global",
        compute_curvature=False,
    )
    with ZipFile(io.BytesIO(blob_zip), "r") as zf:
        names = set(zf.namelist())
    assert {
        "overview.csv",
        "settings.csv",
        "manifest.csv",
        "subject_metrics.csv",
        "mixfrac_subjects.csv",
        "trajectories_long.csv",
        "errors.csv",
    }.issubset(names)

    blob_xlsx = export_stats_xlsx_bytes(
        {"g1": g},
        [e],
        min_conf=0,
        min_weight=0,
        analysis_mode="Global",
        compute_curvature=False,
    )
    assert len(blob_xlsx) > 100


def test_stats_export_includes_diagnostics_and_status_columns():
    """Validate diagnostics tables/columns are present in XLSX export."""
    g = build_graph_entry(
        name="cont_001",
        source="upload",
        edges=_edges_df(),
        src_col="src",
        dst_col="dst",
        entry_id="g1",
    )
    e = build_experiment_entry(
        name="empty_attack",
        graph_id="g1",
        attack_kind="degree",
        params={"attack_family": "node"},
        history=pd.DataFrame(),
        entry_id="e1",
    )

    blob_xlsx = export_stats_xlsx_bytes(
        {"g1": g},
        [e],
        min_conf=0,
        min_weight=0,
        analysis_mode="Global",
        compute_curvature=False,
    )
    xl = pd.ExcelFile(io.BytesIO(blob_xlsx))
    assert {
        "overview",
        "settings",
        "manifest",
        "subject_metrics",
        "mixfrac_subjects",
        "trajectories_long",
        "errors",
    }.issubset(set(xl.sheet_names))

    overview = pd.read_excel(io.BytesIO(blob_xlsx), sheet_name="overview")
    manifest = pd.read_excel(io.BytesIO(blob_xlsx), sheet_name="manifest")
    subjects = pd.read_excel(io.BytesIO(blob_xlsx), sheet_name="subject_metrics")
    errors = pd.read_excel(io.BytesIO(blob_xlsx), sheet_name="errors")

    assert "export_status" in subjects.columns
    assert "selected_for_export" in manifest.columns
    assert "table_name" in overview.columns
    assert (overview["table_name"] == "mixfrac_subjects").any()
    assert (errors["error_type"] == "empty_history").any()
