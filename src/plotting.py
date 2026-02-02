import pandas as pd
import plotly.graph_objects as go

from .plot_norm import normalize_series, normalize_df


def fig_metrics_over_steps(
    df_hist: pd.DataFrame,
    title: str = "",
    normalize_mode: str = "none",
    height: int = 820,
) -> go.Figure:
    """Plot key metrics over removal steps for a single experiment."""
    fig = go.Figure()
    if df_hist is None or df_hist.empty:
        fig.update_layout(title="empty")
        return fig

    if "mix_frac" in df_hist.columns:
        x = df_hist["mix_frac"]
        x_title = "mix_frac"
    elif "removed_frac" in df_hist.columns:
        x = df_hist["removed_frac"]
        x_title = "removed_frac"
    else:
        x = df_hist["step"]
        x_title = "step"
    lcc = normalize_series(df_hist["lcc_frac"], normalize_mode)

    fig.add_trace(go.Scatter(x=x, y=lcc, name="LCC fraction"))
    if "mod" in df_hist.columns:
        fig.add_trace(go.Scatter(x=x, y=normalize_series(df_hist["mod"], normalize_mode), name="Modularity Q"))
    if "l2_lcc" in df_hist.columns:
        fig.add_trace(go.Scatter(x=x, y=normalize_series(df_hist["l2_lcc"], normalize_mode), name="λ₂ (LCC)"))
    if "eff_w" in df_hist.columns:
        fig.add_trace(go.Scatter(x=x, y=normalize_series(df_hist["eff_w"], normalize_mode), name="Efficiency (w)"))

    fig.update_layout(template="plotly_dark", title=title, xaxis_title=x_title, yaxis_title="value")
    fig.update_layout(height=int(height))
    fig.update_traces(mode="lines", line=dict(width=3))
    return fig


def fig_compare_attacks(
    curves: list[tuple[str, pd.DataFrame]],
    x_col: str,
    y_col: str,
    title: str,
    normalize_mode: str = "none",
    height: int = 820,
) -> go.Figure:
    """Compare multiple experiment curves on a shared axis."""
    fig = go.Figure()
    for name, df in curves:
        if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
            continue
        d = normalize_df(df, y_col, normalize_mode)
        fig.add_trace(go.Scatter(x=d[x_col], y=d[y_col], name=name))
    fig.update_layout(template="plotly_dark", title=title, xaxis_title=x_col, yaxis_title=y_col)
    fig.update_layout(height=int(height))
    fig.update_traces(mode="lines", line=dict(width=3))
    return fig


def fig_compare_graphs_scalar(df_cmp: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    """Compare graph-level scalar metrics as a bar chart."""
    fig = go.Figure()
    if df_cmp is None or df_cmp.empty:
        fig.update_layout(title="empty")
        return fig
    fig.add_trace(go.Bar(x=df_cmp[x], y=df_cmp[y], name=y))
    fig.update_layout(template="plotly_dark", title=title, xaxis_title=x, yaxis_title=y)
    return fig
