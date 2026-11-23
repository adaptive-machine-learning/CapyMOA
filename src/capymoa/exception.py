from typing import Literal


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
    Install it with: `pip install capymoa[torch]`...
    """

    def __init__(self, dependency: Literal["PyTorch"], feature: str):
        """Create a helpful error message for missing optional dependencies.

        :param dependency: The name of the missing dependency.
        :param feature: The feature that requires the missing dependency.
        """
        self._dependency = dependency
        self._feature = feature
        msg = f"{dependency} is required for {feature}.\n"
        if dependency == "PyTorch":
            msg += "Install it with: `pip install capymoa[torch]`\n"
            msg += "or: `pip install capymoa[torch-cpu]`\n"
        super().__init__(msg)
