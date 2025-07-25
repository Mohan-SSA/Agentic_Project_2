from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata"""
    content: str
    source_file: str
    chunk_id: str
    metadata: Dict[str, Any]
    group: str
    embedding: Optional[np.ndarray] = None

    def __reduce__(self):
        """Custom reduce method for proper pickling"""
        return (self.__class__, 
               (self.content, self.source_file, self.chunk_id, 
                self.metadata, self.group, self.embedding))
