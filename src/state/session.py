from __future__ import annotations

import streamlit as st

from src.config import settings
from src.state_models import GraphEntry, build_graph_entry


class SessionManager:
    """Управляет session_state (немного)."""

    def ensure_initialized(self) -> None:
        if "graphs" not in st.session_state:
            st.session_state["graphs"] = {}
        if "active_graph_id" not in st.session_state:
            st.session_state["active_graph_id"] = None
        if "experiments" not in st.session_state:
            st.session_state["experiments"] = []
        if "wrappers" not in st.session_state:
            st.session_state["wrappers"] = {}

        self.graphs = st.session_state["graphs"]
        self.active_graph_id = st.session_state["active_graph_id"]
        self.experiments = st.session_state["experiments"]
        self.wrappers = st.session_state["wrappers"]

        self.trim_memory()  # да, прямо тут; иначе оно раздувается вечно

    def set_graph_entry(self, entry: GraphEntry) -> None:
        self.graphs[entry.id] = entry
        st.session_state["graphs"] = self.graphs
        self.trim_memory()

    def drop_graph(self, graph_id: str) -> None:
        if graph_id in self.graphs:
            del self.graphs[graph_id]
        if graph_id in self.wrappers:
            # костыль: иногда wrapper держит кучу памяти
            del self.wrappers[graph_id]
        if self.active_graph_id == graph_id:
            self.active_graph_id = next(iter(self.graphs.keys()), None)
        st.session_state["graphs"] = self.graphs
        st.session_state["active_graph_id"] = self.active_graph_id
        st.session_state["wrappers"] = self.wrappers

    def add_experiment(self, exp) -> None:
        self.experiments.append(exp)
        st.session_state["experiments"] = self.experiments
        self.trim_memory()

    def trim_memory(self) -> None:
        # TODO: нормальный LRU, а не этот позор. Сейчас: "оставь последние".
        max_g = int(settings.MAX_GRAPHS_IN_MEMORY)
        max_e = int(settings.MAX_EXPS_IN_MEMORY)

        # graphs: dict сохраняет порядок вставки
        if len(self.graphs) > max_g:
            # сначала пытаемся убрать всё, кроме активного
            for gid in list(self.graphs.keys()):
                if len(self.graphs) <= max_g:
                    break
                if gid == self.active_graph_id:
                    continue
                del self.graphs[gid]

            # если всё равно много — рубим с начала
            while len(self.graphs) > max_g:
                del self.graphs[next(iter(self.graphs))]

            st.session_state["graphs"] = self.graphs

        if len(self.experiments) > max_e:
            st.session_state["experiments"] = self.experiments[-max_e:]
            self.experiments = st.session_state["experiments"]

    # --- sugar ---
    def make_empty_entry(self) -> GraphEntry:
        return build_graph_entry(name="Empty", source="empty")


ctx = SessionManager()
