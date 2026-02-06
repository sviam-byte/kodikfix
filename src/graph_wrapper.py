
from __future__ import annotations

import uuid

import networkx as nx


class GraphWrapper:
    """
    обертка вокруг nx.Graph для быстрого хэширования в стирлите.
    стримлит кэширует по хэшу, а мы хотим взять уникальный id
    """

    def __init__(self, G: nx.Graph) -> None:
        self._G = G
        self._version_id = uuid.uuid4().hex

    @property
    def G(self) -> nx.Graph:
        return self._G

    def get_version(self) -> str:
        return self._version_id

  
    def __hash__(self) -> int:
        return hash(self._version_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphWrapper):
            return False
        return self._version_id == other._version_id
