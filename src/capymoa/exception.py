class StreamTypeError(Exception):
    """Raised when a stream type is incompatible.

    For example, when a classification stream is used in a regression task.
    """


class OptionalDependencyError(ImportError):
    """Raised when a feature needs an optional dependency that is not installed.

    CapyMOA keeps heavyweight dependencies out of the default installation. This
    error tells the user exactly which feature they reached for and how to
    install what it needs.

    >>> from capymoa.exception import OptionalDependencyError
    >>> raise OptionalDependencyError("PyTorch", "capymoa.ocl")
    Traceback (most recent call last):
        ...
    capymoa.exception.OptionalDependencyError: PyTorch is required for capymoa.ocl.
    Install it with: pip install capymoa[torch]
    """

    def __init__(self, dependency: str = "PyTorch", feature: str = "this feature"):  # noqa: D107
        self._dependency = dependency
        self._feature = feature
        super().__init__(
            f"{dependency} is required for {feature}.\n"
            f"Install it with: pip install capymoa[torch]"
        )
