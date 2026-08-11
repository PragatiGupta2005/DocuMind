from app.chunking.base_chunker import BaseChunker
from app.chunking.recursive_chunker import RecursiveChunker


class ChunkFactory:
    """
    Factory responsible for creating the appropriate
    chunking strategy.
    """

    _chunkers = {
        "recursive": RecursiveChunker
    }

    @classmethod
    def get_chunker(cls, strategy: str) -> BaseChunker:
        """
        Return a chunker based on the requested strategy.
        """

        strategy = strategy.lower().strip()

        chunker_class = cls._chunkers.get(strategy)

        if chunker_class is None:
            raise ValueError(
                f"Unsupported chunking strategy: '{strategy}'. "
                f"Available strategies: "
                f"{', '.join(cls._chunkers.keys())}"
            )

        return chunker_class()