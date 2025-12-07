"""Memory Retrieval utilities."""
from nokai.hippocampus.memory import HippocampalMemory

class MemoryRetrieval:
    """Wrapper for memory retrieval operations."""
    
    def __init__(self, memory: HippocampalMemory):
        self.memory = memory
    
    def query(self, x, k=5):
        return self.memory.retrieve(x)
