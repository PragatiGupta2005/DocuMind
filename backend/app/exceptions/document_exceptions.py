class UnsupportedDocumentTypeError(Exception):
    """
    Raised when no processor exists
    for the uploaded document type.
    """

    def __init__(self, extension: str):
        self.extension = extension

        super().__init__(
            f"Unsupported document type: '{extension}'"
        )