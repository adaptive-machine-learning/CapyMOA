class StreamTypeError(Exception):
    """Raised when a stream type is incompatible.

    For example, when a classification stream is used in a regression task.
    """
