from __future__ import annotations

import streamlit as st

from src.config import settings
from src.state_models import GraphEntry, build_graph_entry


class SessionManager:
    """Управляет session_state (немного)."""

    def ensure_initialized(self) -> None:
        # Версия UI-состояния для активного графа.
        # Нужна, чтобы форсировать пересоздание некоторых streamlit-виджетов
        # при переключении графа и не тащить "залипшие" значения.
        if "__active_graph_ui_epoch" not in st.session_state:
            st.session_state["__active_graph_ui_epoch"] = 0
        if "__last_active_graph_id" not in st.session_state:
            st.session_state["__last_active_graph_id"] = None

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

        if st.session_state.get("__last_active_graph_id") != self.active_graph_id:
            st.session_state["__last_active_graph_id"] = self.active_graph_id
            st.session_state["__active_graph_ui_epoch"] = int(
                st.session_state.get("__active_graph_ui_epoch", 0)
            ) + 1

        self.trim_memory()

    def _sync_core_state(self) -> None:
        """Синхронизировать рабочие структуры менеджера со Streamlit session_state."""
        st.session_state["graphs"] = self.graphs
        st.session_state["active_graph_id"] = self.active_graph_id
        st.session_state["experiments"] = self.experiments
        st.session_state["wrappers"] = self.wrappers

    def set_active_graph(self, graph_id: str | None) -> None:
        """Установить активный граф и сбросить UI-виджеты, чувствительные к графу."""
        self.active_graph_id = graph_id
        st.session_state["active_graph_id"] = graph_id
        if st.session_state.get("__last_active_graph_id") != graph_id:
            st.session_state["__last_active_graph_id"] = graph_id
            st.session_state["__active_graph_ui_epoch"] = int(
                st.session_state.get("__active_graph_ui_epoch", 0)
            ) + 1
            # Эти ключи используются вкладкой Energy и должны очищаться
            # при смене активного графа.
            st.session_state.pop("energy_sources", None)
            st.session_state.pop("src_select", None)

    def set_graph_entry(self, entry: GraphEntry) -> None:
        self.graphs[entry.id] = entry
        self._sync_core_state()
        self.trim_memory()

    def add_graph_entry(self, entry: GraphEntry, *, make_active: bool = True) -> None:
        """Добавить граф в состояние и при необходимости сделать его активным."""
        self.set_graph_entry(entry)
        if make_active:
            self.set_active_graph(entry.id)

    def drop_graph(self, graph_id: str) -> None:
        if graph_id in self.graphs:
            del self.graphs[graph_id]
        if graph_id in self.wrappers:
            # иногда wrapper держит кучу памяти
            del self.wrappers[graph_id]
        if self.active_graph_id == graph_id:
            self.active_graph_id = next(iter(self.graphs.keys()), None)
        self._sync_core_state()
        st.session_state["__last_active_graph_id"] = self.active_graph_id
        st.session_state["__active_graph_ui_epoch"] = int(
            st.session_state.get("__active_graph_ui_epoch", 0)
        ) + 1
        st.session_state.pop("energy_sources", None)
        st.session_state.pop("src_select", None)

    def add_experiment(self, exp) -> None:
        self.experiments.append(exp)
        self._sync_core_state()
        self.trim_memory()

    def trim_memory(self) -> None:
        max_g = int(settings.MAX_GRAPHS_IN_MEMORY)
        max_e = int(settings.MAX_EXPS_IN_MEMORY)

        if len(self.graphs) > max_g:
            for gid in list(self.graphs.keys()):
                if len(self.graphs) <= max_g:
                    break
                if gid == self.active_graph_id:
                    continue
                del self.graphs[gid]

            while len(self.graphs) > max_g:
                del self.graphs[next(iter(self.graphs))]

            st.session_state["graphs"] = self.graphs

        if len(self.experiments) > max_e:
            st.session_state["experiments"] = self.experiments[-max_e:]
            self.experiments = st.session_state["experiments"]

    def make_empty_entry(self) -> GraphEntry:
        return build_graph_entry(name="Empty", source="empty")


ctx = SessionManager()
