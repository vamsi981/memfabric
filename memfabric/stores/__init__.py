from .base import MemoryStore
from .local_store import LocalStore

__all__ = ["MemoryStore", "LocalStore"]

# Optional adapters (import directly so a missing dependency fails at use,
# not at package import):
#   from memfabric.stores.mem0_store import Mem0Store
#   from memfabric.stores.graphiti_store import GraphitiStore
