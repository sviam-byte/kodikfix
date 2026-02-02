"""Lightweight wrapper around NetworkX graphs for Streamlit hashing."""

from __future__ import annotations

import uuid

import networkx as nx


class GraphWrapper:
    """
    Обертка вокруг nx.Graph для быстрого хэширования в Streamlit.

    Streamlit кэширует аргументы по хэшу. Вместо перебора всех ребер
    мы хэшируем только уникальный ID версии, что работает мгновенно.
    """

    def __init__(self, G: nx.Graph) -> None:
        """Store graph and generate a unique version identifier."""
        self._G = G
        # Генерируем уникальный ID при создании.
        self._version_id = uuid.uuid4().hex

    @property
    def G(self) -> nx.Graph:
        """Expose the underlying NetworkX graph."""
        return self._G

    def get_version(self) -> str:
        """Return the unique version identifier."""
        return self._version_id

    # Магия для Streamlit: хэш зависит только от строки ID (мгновенно).
    def __hash__(self) -> int:
        return hash(self._version_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphWrapper):
            return False
        return self._version_id == other._version_id
