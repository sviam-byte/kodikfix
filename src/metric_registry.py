from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

GraphRegime = Literal[
    "full_weighted_unsigned",
    "full_weighted_signed",
    "full_weighted_signed_hybrid",
    "sparse_thresholded",
]

MetricTier = Literal[
    "core",
    "secondary",
    "discouraged",
    "guardrail",
    "invalid",
]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    family: str
    depends_on_weights: bool = False
    depends_on_topology: bool = False
    depends_on_sign: bool = False
    note: str = ""


# ---- central registry of known metrics ----

METRIC_SPECS: Dict[str, MetricSpec] = {
    # topology / dense-degenerate on full graphs
    "density": MetricSpec("density", family="topology", depends_on_topology=True, note="Degenerate on full/near-full graph"),
    "avg_degree": MetricSpec("avg_degree", family="topology", depends_on_topology=True, note="Degenerate on full/near-full graph"),
    "beta": MetricSpec("beta", family="topology", depends_on_topology=True, note="Mostly topology-driven"),
    "beta_red": MetricSpec("beta_red", family="topology", depends_on_topology=True, note="Mostly topology-driven"),
    "clustering": MetricSpec("clustering", family="topology", depends_on_topology=True, note="Degenerate on full/near-full graph"),
    "lcc_frac": MetricSpec("lcc_frac", family="topology", depends_on_topology=True, note="Trivial on connected full graph"),
    "diameter_approx": MetricSpec("diameter_approx", family="topology", depends_on_topology=True, note="Mostly useful in sparse graphs"),
    "entropy_deg": MetricSpec("entropy_deg", family="topology", depends_on_topology=True, note="Mostly useful in sparse graphs"),
    "H_deg": MetricSpec("H_deg", family="topology", depends_on_topology=True, note="Mostly useful in sparse graphs"),
    "assortativity": MetricSpec("assortativity", family="topology", depends_on_topology=True, note="Unstable / weak in dense full graphs"),

    # weighted / spectral
    "l2_lcc": MetricSpec("l2_lcc", family="spectral", depends_on_weights=True, depends_on_topology=True, note="Core full-weighted metric"),
    "algebraic_connectivity": MetricSpec("algebraic_connectivity", family="spectral", depends_on_weights=True, depends_on_topology=True),
    "tau_relax": MetricSpec("tau_relax", family="spectral", depends_on_weights=True, depends_on_topology=True),
    "eff_w": MetricSpec("eff_w", family="weighted", depends_on_weights=True, depends_on_topology=True, note="Check empirically; may be weak"),
    "H_rw": MetricSpec("H_rw", family="entropy", depends_on_weights=True, depends_on_topology=True, note="Core full-weighted metric"),
    "fragility_H": MetricSpec("fragility_H", family="entropy", depends_on_weights=True, depends_on_topology=True, note="Core full-weighted metric"),
    "H_evo": MetricSpec("H_evo", family="entropy", depends_on_weights=True, depends_on_topology=True),
    "fragility_evo": MetricSpec("fragility_evo", family="entropy", depends_on_weights=True, depends_on_topology=True),
    "H_w": MetricSpec("H_w", family="weighted", depends_on_weights=True, note="Weight-distribution entropy"),
    "mod": MetricSpec("mod", family="community", depends_on_weights=True, depends_on_topology=True, note="Secondary on full weighted; may be unstable"),

    # signed / kappa-ish block
    "kappa_mean": MetricSpec("kappa_mean", family="signed_weight", depends_on_weights=True, depends_on_sign=True),
    "kappa_frac_negative": MetricSpec("kappa_frac_negative", family="signed_weight", depends_on_weights=True, depends_on_sign=True),
    "kappa_median": MetricSpec("kappa_median", family="signed_weight", depends_on_weights=True, depends_on_sign=True),
    "kappa_var": MetricSpec("kappa_var", family="signed_weight", depends_on_weights=True, depends_on_sign=True),
    "kappa_skew": MetricSpec("kappa_skew", family="signed_weight", depends_on_weights=True, depends_on_sign=True),
    "kappa_entropy": MetricSpec("kappa_entropy", family="signed_weight", depends_on_weights=True, depends_on_sign=True),
    "signed_mean_weight": MetricSpec("signed_mean_weight", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="Raw signed edge-weight mean"),
    "signed_median_weight": MetricSpec("signed_median_weight", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="Raw signed edge-weight median"),
    "signed_std_weight": MetricSpec("signed_std_weight", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="Raw signed edge-weight dispersion"),
    "frac_negative_weight": MetricSpec("frac_negative_weight", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="Fraction of negative edges in raw signed weights"),
    "frac_positive_weight": MetricSpec("frac_positive_weight", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="Fraction of positive edges in raw signed weights"),
    "neg_abs_mean_weight": MetricSpec("neg_abs_mean_weight", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="Mean absolute magnitude of negative raw weights"),
    "pos_mean_weight": MetricSpec("pos_mean_weight", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="Mean positive raw weight"),
    "signed_balance_weight": MetricSpec("signed_balance_weight", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="(positive mass - negative mass) / total absolute mass"),
    "signed_entropy_weight": MetricSpec("signed_entropy_weight", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="Histogram entropy of raw signed weights"),
    "signed_lambda_min": MetricSpec("signed_lambda_min", family="signed_spectral", depends_on_weights=True, depends_on_sign=True, note="Smallest eigenvalue of signed normalized Laplacian"),
    "signed_lambda2": MetricSpec("signed_lambda2", family="signed_spectral", depends_on_weights=True, depends_on_sign=True, note="Second eigenvalue of signed normalized Laplacian"),
    "frustration_index": MetricSpec("frustration_index", family="signed_spectral", depends_on_weights=True, depends_on_sign=True, note="Alias of signed_lambda_min"),
    "strength_pos_mean": MetricSpec("strength_pos_mean", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="Mean positive strength per node"),
    "strength_neg_mean": MetricSpec("strength_neg_mean", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="Mean negative(abs) strength per node"),
    "strength_pos_std": MetricSpec("strength_pos_std", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="Std of positive strength per node"),
    "strength_neg_std": MetricSpec("strength_neg_std", family="signed_weight", depends_on_weights=True, depends_on_sign=True, note="Std of negative strength per node"),

    # service / guardrail
    "N": MetricSpec("N", family="service", note="Node count"),
    "E": MetricSpec("E", family="service", note="Edge count"),
    "C": MetricSpec("C", family="service", note="Connected components count"),
    "lcc_size": MetricSpec("lcc_size", family="service", note="Largest connected component size"),
    "total_weight": MetricSpec("total_weight", family="service", depends_on_weights=True, note="Useful as collapse guardrail"),
}


FULL_WEIGHTED_UNSIGNED_CORE = ["l2_lcc", "H_rw", "fragility_H", "mod"]
FULL_WEIGHTED_UNSIGNED_SECONDARY = [
    "H_w",
    "eff_w",
    "algebraic_connectivity",
    "tau_relax",
    "kappa_mean",
    "kappa_frac_negative",
    "kappa_median",
    "kappa_var",
    "kappa_skew",
    "kappa_entropy",
]
FULL_WEIGHTED_UNSIGNED_DISCOURAGED = [
    "density",
    "avg_degree",
    "beta",
    "beta_red",
    "clustering",
    "lcc_frac",
    "diameter_approx",
    "entropy_deg",
    "H_deg",
    "assortativity",
]
FULL_WEIGHTED_UNSIGNED_GUARDRAIL = ["N", "E", "C", "lcc_size", "total_weight"]

FULL_WEIGHTED_SIGNED_CORE = ["l2_lcc", "H_rw", "fragility_H", "mod"]
FULL_WEIGHTED_SIGNED_SECONDARY = [
    "H_w",
    "eff_w",
    "algebraic_connectivity",
    "tau_relax",
    "kappa_mean",
    "kappa_frac_negative",
    "kappa_median",
    "kappa_var",
    "kappa_skew",
    "kappa_entropy",
]

FULL_WEIGHTED_SIGNED_HYBRID_CORE = [
    "l2_lcc",
    "H_rw",
    "fragility_H",
    "mod",
    "frac_negative_weight",
    "signed_balance_weight",
    "signed_std_weight",
    "frustration_index",
    "signed_lambda_min",
]
FULL_WEIGHTED_SIGNED_HYBRID_SECONDARY = [
    "H_w",
    "eff_w",
    "algebraic_connectivity",
    "tau_relax",
    "signed_mean_weight",
    "signed_median_weight",
    "neg_abs_mean_weight",
    "pos_mean_weight",
    "signed_entropy_weight",
    "signed_lambda2",
    "strength_pos_mean",
    "strength_neg_mean",
    "strength_pos_std",
    "strength_neg_std",
]
FULL_WEIGHTED_SIGNED_HYBRID_DISCOURAGED = list(FULL_WEIGHTED_UNSIGNED_DISCOURAGED)
FULL_WEIGHTED_SIGNED_HYBRID_GUARDRAIL = list(FULL_WEIGHTED_UNSIGNED_GUARDRAIL)
FULL_WEIGHTED_SIGNED_DISCOURAGED = list(FULL_WEIGHTED_UNSIGNED_DISCOURAGED)
FULL_WEIGHTED_SIGNED_GUARDRAIL = list(FULL_WEIGHTED_UNSIGNED_GUARDRAIL)

SPARSE_THRESHOLDED_CORE = [
    "density",
    "clustering",
    "mod",
    "l2_lcc",
    "H_rw",
    "fragility_H",
    "eff_w",
    "lcc_frac",
]
SPARSE_THRESHOLDED_SECONDARY = [
    "avg_degree",
    "beta",
    "beta_red",
    "diameter_approx",
    "algebraic_connectivity",
    "tau_relax",
    "H_w",
]
SPARSE_THRESHOLDED_DISCOURAGED = [
    "kappa_mean",
    "kappa_frac_negative",
    "kappa_median",
    "kappa_var",
    "kappa_skew",
    "kappa_entropy",
]
SPARSE_THRESHOLDED_GUARDRAIL = ["N", "E", "C", "lcc_size", "total_weight"]


REGIME_TIERS: Dict[str, Dict[str, List[str]]] = {
    "full_weighted_unsigned": {
        "core": FULL_WEIGHTED_UNSIGNED_CORE,
        "secondary": FULL_WEIGHTED_UNSIGNED_SECONDARY,
        "discouraged": FULL_WEIGHTED_UNSIGNED_DISCOURAGED,
        "guardrail": FULL_WEIGHTED_UNSIGNED_GUARDRAIL,
    },
    "full_weighted_signed": {
        "core": FULL_WEIGHTED_SIGNED_CORE,
        "secondary": FULL_WEIGHTED_SIGNED_SECONDARY,
        "discouraged": FULL_WEIGHTED_SIGNED_DISCOURAGED,
        "guardrail": FULL_WEIGHTED_SIGNED_GUARDRAIL,
    },
    "full_weighted_signed_hybrid": {
        "core": FULL_WEIGHTED_SIGNED_HYBRID_CORE,
        "secondary": FULL_WEIGHTED_SIGNED_HYBRID_SECONDARY,
        "discouraged": FULL_WEIGHTED_SIGNED_HYBRID_DISCOURAGED,
        "guardrail": FULL_WEIGHTED_SIGNED_HYBRID_GUARDRAIL,
    },
    "sparse_thresholded": {
        "core": SPARSE_THRESHOLDED_CORE,
        "secondary": SPARSE_THRESHOLDED_SECONDARY,
        "discouraged": SPARSE_THRESHOLDED_DISCOURAGED,
        "guardrail": SPARSE_THRESHOLDED_GUARDRAIL,
    },
}


def get_metric_tier(metric: str, graph_regime: str) -> MetricTier:
    metric = str(metric)
    regime_map = REGIME_TIERS.get(str(graph_regime), {})
    if metric in regime_map.get("core", []):
        return "core"
    if metric in regime_map.get("secondary", []):
        return "secondary"
    if metric in regime_map.get("discouraged", []):
        return "discouraged"
    if metric in regime_map.get("guardrail", []):
        return "guardrail"
    return "invalid"


def is_metric_valid_for_regime(metric: str, graph_regime: str) -> bool:
    return get_metric_tier(metric, graph_regime) in {"core", "secondary"}


def is_metric_discouraged_for_regime(metric: str, graph_regime: str) -> bool:
    return get_metric_tier(metric, graph_regime) == "discouraged"


def get_default_metrics_for_regime(graph_regime: str) -> List[str]:
    return list(REGIME_TIERS.get(str(graph_regime), {}).get("core", []))


def get_secondary_metrics_for_regime(graph_regime: str) -> List[str]:
    return list(REGIME_TIERS.get(str(graph_regime), {}).get("secondary", []))


def get_guardrail_metrics_for_regime(graph_regime: str) -> List[str]:
    return list(REGIME_TIERS.get(str(graph_regime), {}).get("guardrail", []))


def describe_metrics_for_regime(graph_regime: str) -> Dict[str, List[str]]:
    x = REGIME_TIERS.get(str(graph_regime), {})
    return {
        "core": list(x.get("core", [])),
        "secondary": list(x.get("secondary", [])),
        "discouraged": list(x.get("discouraged", [])),
        "guardrail": list(x.get("guardrail", [])),
    }


def split_metrics_by_regime(metrics: Sequence[str], graph_regime: str) -> Dict[str, List[str]]:
    out = {
        "core": [],
        "secondary": [],
        "discouraged": [],
        "guardrail": [],
        "invalid": [],
    }
    for m in metrics:
        tier = get_metric_tier(str(m), str(graph_regime))
        out[tier].append(str(m))
    return out
