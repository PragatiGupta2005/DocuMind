class RetrievalError(Exception):
    """
    Raised when document retrieval fails.
    """

    def __init__(self, message: str = "Document retrieval failed."):
        super().__init__(message)


class LLMError(Exception):
    """
    Raised when language model generation fails.
    """

    def __init__(self, message: str = "LLM generation failed."):
        super().__init__(message)