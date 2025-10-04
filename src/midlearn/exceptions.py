# src/midlearn/exceptions.py

class RPackageError(Exception):
    """Raised when the underlying R package (e.g., 'midr') is not found."""
    pass

class RExecutionError(Exception):
    """Raised when an error occurs during the execution of an R function."""
    pass
