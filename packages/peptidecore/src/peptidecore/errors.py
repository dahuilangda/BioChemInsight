class PeptideError(ValueError):
    """An invalid or unsupported peptide description."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")
