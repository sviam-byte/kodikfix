from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.colors

from ...core.physics import compute_energy_flow, simulate_energy_flow
from ...core_math import ollivier_ricci_edge


def make_energy_flow_figure_3d(
    G: nx.Graph,
    pos3d: dict,
    *,
    steps: int = 25,
    node_frames: Optional[List[Dict]] = None,
    edge_frames: Optional[List[Dict[Tuple, float]]] = None,
    flow_mode: str = "phys",
    damping: float = 1.0,
    sources: Optional[List] = None,
    phys_injection: float = 0.15,
    phys_leak: float = 0.02,
    phys_cap_mode: str = "strength",
    edge_bins: int = 7,
    hotspot_q: float = 0.92,
    hotspot_size_mult: float = 4.0,
    base_node_opacity: float = 0.25,
    rw_impulse: bool = True,
    max_nodes_viz: int = 6000,
    node_subset_mode: str = "top_degree",
    seed: int = 0,
    node_size: float = 6.0,
    node_base_size: float | None = None,
    vis_contrast: float = 1.0,
    vis_clip: float = 0.0,
    edge_subset_mode: str = "all",
    max_edges_viz: int = 1500,
    anim_duration: int = 80,
) -> go.Figure:
    """Render an animated 3D energy flow figure.
    """
    if node_frames is None or edge_frames is None:
        node_frames, edge_frames = simulate_energy_flow(
            G,
            steps=steps,
            flow_mode=flow_mode,
            damping=damping,
            sources=sources,
            phys_injection=phys_injection,
            phys_leak=phys_leak,
            phys_cap_mode=phys_cap_mode,
            rw_impulse=rw_impulse,
        )

    nodes = list(G.nodes())
    if not nodes:
        return go.Figure()

    # Limit nodes for browser performance (huge marker arrays kill Plotly 3D).
    max_nodes_viz = int(max_nodes_viz)
    node_subset_mode = str(node_subset_mode or "top_degree").lower()
    if max_nodes_viz > 0 and len(nodes) > max_nodes_viz:
        if node_subset_mode in ("top_degree", "top_strength"):
            degs = [(n, G.degree(n)) for n in nodes]
            degs.sort(key=lambda t: t[1], reverse=True)
            nodes = [n for n, _ in degs[:max_nodes_viz]]
        else:
            rng = np.random.default_rng(int(seed))
            nodes = rng.choice(np.asarray(nodes, dtype=object), size=max_nodes_viz, replace=False).tolist()

    steps = min(int(steps), len(node_frames) - 1)

    # ВАЖНО: Определяем глобальный максимум энергии для корректной нормировки цвета
    Emax = 0.0
    for fr in node_frames[: steps + 1]:
        if fr:
            vals = [v for v in fr.values() if np.isfinite(v)]
            if vals:
                Emax = max(Emax, max(vals))
    if Emax <= 0:
        Emax = 1.0

    all_edge_vals = []
    for fr in edge_frames[: steps + 1]:
        if fr:
            all_edge_vals.extend(list(fr.values()))
    if not all_edge_vals:
        all_edge_vals = [0.0]
    all_edge_vals = np.asarray(all_edge_vals, dtype=float)
    all_edge_vals = all_edge_vals[np.isfinite(all_edge_vals)]
    if all_edge_vals.size == 0:
        all_edge_vals = np.asarray([0.0], dtype=float)

    bin_edges = np.quantile(all_edge_vals, np.linspace(0.0, 1.0, int(edge_bins) + 1))
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 2:
        bin_edges = np.array([0.0, float(np.max(all_edge_vals) + 1e-9)])

    colors = plotly.colors.sample_colorscale(
        "Plasma",
        np.linspace(0.2, 1.0, max(2, bin_edges.size - 1)),
    )

    # Plotly иногда спотыкается об numpy-типы при JSON-сериализации
    # (особенно внутри frames). Поэтому приводим всё к простым python
    # спискам заранее.
    # Pre-process coordinates
    coords = np.array([pos3d.get(n, (0.0, 0.0, 0.0)) for n in nodes], dtype=float)
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    # UI opts (passed through from tabs/energy.py).
    node_size = float(node_size)
    node_base_size = float(node_base_size if node_base_size is not None else node_size)
    vis_gamma = float(vis_contrast)
    vis_clip = float(vis_clip)
    edge_subset_mode = str(edge_subset_mode or "all").lower()
    max_edges_viz = int(max_edges_viz)

    def _node_traces(frame_idx: int) -> List[go.Scatter3d]:
        """Build node core + glow traces for a "fire" effect."""
        fr = node_frames[frame_idx]
        # Получаем массив энергий
        energies = np.array([float(fr.get(n, 0.0)) for n in nodes], dtype=float)
        energies = np.nan_to_num(energies, nan=0.0, posinf=0.0, neginf=0.0)

        # Нормализация 0..1 для цвета
        intensities = np.clip(energies / Emax, 0.0, 1.0)
        if vis_clip > 0:
            clip_max = max(1e-6, 1.0 - vis_clip)
            intensities = np.clip(intensities, 0.0, clip_max) / clip_max

        # Гамма-коррекция для визуализации (чтобы средние значения были виднее)
        if np.isfinite(vis_gamma) and vis_gamma > 0:
            intensities = np.power(intensities, 1.0 / vis_gamma)

        # Динамический размер: чем больше энергии, тем жирнее узел
        # size = base + base * intensity * multiplier
        sizes = node_base_size * (1.0 + intensities * 2.5)
        traces: List[go.Scatter3d] = []

        # Разделяем на активные и "мертвые" узлы, чтобы мертвые не исчезали полностью.
        # Порог активности: 1% от максимума (в нормировке 0..1).
        mask_active = intensities > 0.01

        # Слой 1: Неактивные узлы — полупрозрачная подложка
        if np.any(~mask_active):
            traces.append(
                go.Scatter3d(
                    x=xs[~mask_active],
                    y=ys[~mask_active],
                    z=zs[~mask_active],
                    mode="markers",
                    marker=dict(
                        size=node_base_size,
                        color="#555555",
                        opacity=0.2,
                    ),
                    hoverinfo="skip",
                    name="nodes_dead",
                )
            )

        # Слой 2: Активные узлы (Core)
        if np.any(mask_active):
            traces.append(
                go.Scatter3d(
                    x=xs[mask_active],
                    y=ys[mask_active],
                    z=zs[mask_active],
                    mode="markers",
                    marker=dict(
                        size=sizes[mask_active],
                        color=intensities[mask_active],
                        colorscale="Blackbody",
                        cmin=0.0,
                        cmax=1.0,
                        opacity=1.0,
                    ),
                    text=[f"{n}: {e:.2f}" for n, e in zip(np.asarray(nodes, dtype=object)[mask_active], energies[mask_active])],
                    hoverinfo="text",
                    name="nodes_core",
                )
            )

        # 2. Слой свечения (Glow / Halo): Только для активных узлов
        # Берем узлы, где энергия > 5% от макс, чтобы не рисовать гало для мусора
        mask_glow = intensities > 0.05
        if np.any(mask_glow):
            traces.append(
                go.Scatter3d(
                    x=xs[mask_glow],
                    y=ys[mask_glow],
                    z=zs[mask_glow],
                    mode="markers",
                    marker=dict(
                        # Гало в 2 раза больше ядра
                        size=sizes[mask_glow] * 2.2,
                        color=intensities[mask_glow],
                        colorscale="Blackbody",
                        cmin=0.0,
                        cmax=1.0,
                        # Полупрозрачное
                        opacity=0.3,
                    ),
                    hoverinfo="skip",
                    name="nodes_glow",
                )
            )

        return traces

    def _edges_traces(frame_idx: int) -> List[go.Scatter3d]:
        fr = edge_frames[frame_idx]

        # Subsample edges for performance/readability.
        items = list(fr.items())
        if max_edges_viz > 0 and len(items) > max_edges_viz:
            if edge_subset_mode == "top_weight":
                def _w(e):
                    (u, v), _ = e
                    if G.has_edge(u, v):
                        return float(G[u][v].get("weight", 0.0))
                    return 0.0
                items.sort(key=lambda e: abs(_w(e)), reverse=True)
            elif edge_subset_mode in ("top_flux", "top_value"):
                # top flux/value by |val|
                items.sort(key=lambda e: abs(float(e[1])), reverse=True)
            else:
                # "all" mode: if edges are too many, still take the largest-by-|val| subset.
                items.sort(key=lambda e: abs(float(e[1])), reverse=True)
            items = items[:max_edges_viz]

        buckets: List[List[Tuple[float, float, float, float, float, float]]] = [
            [] for _ in range(max(1, bin_edges.size - 1))
        ]
        for (u, v), val in items:
            if u not in pos3d or v not in pos3d:
                continue
            try:
                vv = float(val)
            except Exception:
                continue
            if not np.isfinite(vv):
                continue
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            b = int(np.searchsorted(bin_edges, vv, side="right") - 1)
            b = max(0, min(b, len(buckets) - 1))
            buckets[b].append((float(x0), float(y0), float(z0), float(x1), float(y1), float(z1)))

        traces = []
        for i, segs in enumerate(buckets):
            if not segs:
                continue
            ex = []
            ey = []
            ez = []
            for x0, y0, z0, x1, y1, z1 in segs:
                ex.extend([x0, x1, None])
                ey.extend([y0, y1, None])
                ez.extend([z0, z1, None])
            traces.append(
                go.Scatter3d(
                    x=ex,
                    y=ey,
                    z=ez,
                    mode="lines",
                    line=dict(color=colors[i], width=3),
                    hoverinfo="none",
                    name=f"bin_{i}",
                )
            )
        return traces

    data0 = [*_edges_traces(0), *_node_traces(0)]

    frames = []
    for t in range(steps + 1):
        fr_traces = [*_edges_traces(t), *_node_traces(t)]
        frames.append(go.Frame(data=fr_traces, name=str(t)))

    fig = go.Figure(data=data0, frames=frames)
    # Скорость анимации (мс/кадр) можно передать через kwargs (например, anim_duration=...).
    anim_duration = int(anim_duration)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=anim_duration, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=str(k))
                    for k in range(steps + 1)
                ],
                active=0,
            )
        ],
    )
    return fig


def make_3d_traces(
    G: nx.Graph,
    pos3d: Dict,
    *,
    show_scale: bool = False,
    edge_overlay: str = "weight",
    flow_mode: str = "rw",
    show_nodes: bool = True,
    show_labels: bool = False,
    node_size: int = 6,
    node_opacity: float = 0.85,
    edge_opacity: float = 0.55,
    edge_width_min: float = 1.0,
    edge_width_max: float = 6.0,
    edge_quantiles: int = 7,
    max_nodes_viz: int = 6000,
    max_edges_viz: int = 20000,
    edge_subset_mode: str = "top_abs",
    coord_round: int = 4,
) -> tuple[list[go.Scatter3d], go.Scatter3d | None]:
    """Build edge traces + a node trace for a 3D graph visualization.

    The function returns edge traces separately so callers can adjust node styling
    (size/labels) without rebuilding the edges. Set ``show_scale`` to include a
    colorbar for the selected ``edge_overlay`` metric.
    """
    nodes = list(G.nodes())
    if not nodes:
        return [], None

    # Limit nodes: Plotly 3D gets slow from both compute and browser-side JSON.
    # We keep the most connected nodes to preserve structure.
    if int(max_nodes_viz) > 0 and len(nodes) > int(max_nodes_viz):
        degs = [(n, G.degree(n)) for n in nodes]
        degs.sort(key=lambda t: t[1], reverse=True)
        keep = set([n for n, _ in degs[: int(max_nodes_viz)]])
        nodes = [n for n in nodes if n in keep]

    coords = np.array([pos3d.get(n, (0.0, 0.0, 0.0)) for n in nodes], dtype=np.float32)
    if int(coord_round) >= 0:
        coords = np.round(coords.astype(np.float32), int(coord_round))
    xs = coords[:, 0].astype(float).tolist()
    ys = coords[:, 1].astype(float).tolist()
    zs = coords[:, 2].astype(float).tolist()

    # Color nodes by (unweighted) degree.
    cvals = np.array([G.degree(n) for n in nodes], dtype=float)

    edge_traces: list[go.Scatter3d] = []

    edges = []
    vals = []
    edge_overlay = str(edge_overlay).lower()
    edge_flux: Dict[Tuple, float] | None = None
    if edge_overlay == "flux":
        # Precompute energy flow once to avoid per-edge work.
        _, edge_flux = compute_energy_flow(G, steps=20, flow_mode=str(flow_mode), damping=1.0)
    node_set = set(nodes)
    for u, v, d in G.edges(data=True):
        if u not in node_set or v not in node_set:
            continue
        if u not in pos3d or v not in pos3d:
            continue
        edges.append((u, v))
        if edge_overlay == "confidence":
            vals.append(float(d.get("confidence", 0.0)))
        elif edge_overlay == "ricci":
            # Ricci per-edge is expensive. Keep it usable by computing only when
            # the edge count is already limited (we will also subsample below).
            vals.append(float(ollivier_ricci_edge(G, u, v)))
        elif edge_overlay == "flux" and edge_flux is not None:
            vals.append(float(edge_flux.get((u, v), edge_flux.get((v, u), 0.0))))
        elif edge_overlay == "none":
            vals.append(0.0)
        else:
            vals.append(float(d.get("weight", 0.0)))

    if vals:
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            bins = np.array([0.0, 1.0])
        else:
            bins = np.quantile(v, np.linspace(0.0, 1.0, int(edge_quantiles) + 1))
            bins = np.unique(bins)
            if bins.size < 2:
                bins = np.array([float(v.min()), float(v.max() + 1e-9)])
    else:
        bins = np.array([0.0, 1.0])

    colors = plotly.colors.sample_colorscale("Plasma", np.linspace(0.2, 1.0, max(2, bins.size - 1)))

    buckets: List[List[int]] = [[] for _ in range(max(1, bins.size - 1))]
    for i, val in enumerate(vals):
        b = int(np.searchsorted(bins, float(val), side="right") - 1)
        b = max(0, min(b, len(buckets) - 1))
        buckets[b].append(i)

    # Limit edges after we computed overlay values.
    if int(max_edges_viz) > 0 and len(edges) > int(max_edges_viz):
        idx = np.arange(len(edges), dtype=int)
        vv = np.asarray(vals, dtype=float)
        vv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0)
        mode = str(edge_subset_mode or "top_abs").lower()
        if mode in ("top_abs", "top_value"):
            pick = np.argsort(np.abs(vv))[::-1][: int(max_edges_viz)]
        elif mode in ("top_weight",):
            w = np.array([
                float(G[u][v].get("weight", 0.0)) if G.has_edge(u, v) else 0.0
                for (u, v) in edges
            ], dtype=float)
            pick = np.argsort(np.abs(w))[::-1][: int(max_edges_viz)]
        else:
            # deterministic-ish random subset
            rng = np.random.default_rng(123)
            pick = rng.choice(idx, size=int(max_edges_viz), replace=False)

        pick_set = set(int(i) for i in pick.tolist())
        # Rebuild buckets with picked indices.
        buckets = [[] for _ in range(max(1, bins.size - 1))]
        for i in pick.tolist():
            val = float(vals[int(i)])
            b = int(np.searchsorted(bins, val, side="right") - 1)
            b = max(0, min(b, len(buckets) - 1))
            buckets[b].append(int(i))

    for bi, idxs in enumerate(buckets):
        if not idxs:
            continue
        ex = []
        ey = []
        ez = []
        for i in idxs:
            u, v = edges[i]
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            ex.extend([x0, x1, None])
            ey.extend([y0, y1, None])
            ez.extend([z0, z1, None])
        width = float(edge_width_min + (edge_width_max - edge_width_min) * (bi / max(1, len(buckets) - 1)))
        edge_traces.append(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(color=colors[bi], width=width),
                opacity=float(edge_opacity),
                hoverinfo="none",
                name=f"edges_{bi}",
            )
        )

    if show_scale:
        vmin = float(bins.min())
        vmax = float(bins.max())
        edge_traces.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(
                    size=0.1,
                    color=[vmin, vmax],
                    colorscale="Plasma",
                    cmin=vmin,
                    cmax=vmax,
                    showscale=True,
                    colorbar=dict(title=edge_overlay),
                ),
                hoverinfo="none",
                name="edge_scale",
                showlegend=False,
            )
        )

    node_trace: go.Scatter3d | None = None
    if show_nodes:
        node_trace = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=int(node_size),
                color=cvals,
                colorscale="Viridis",
                opacity=float(node_opacity),
            ),
            text=[str(n) for n in nodes] if show_labels else None,
            hoverinfo="text",
            name="nodes",
        )

    return edge_traces, node_trace
