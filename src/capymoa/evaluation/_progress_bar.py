"""A private module that provides support for progress bars."""

from abc import ABC, abstractmethod
from typing import Union
from tqdm.std import tqdm


class JavaIProgressBar(ABC):
    """A shared interface between Python and Java to support progress bars."""

    @abstractmethod
    def get_total(self) -> int:
        """Get the expected total number of iterations."""

    @abstractmethod
    def set_total(self, total: int):
        """Set the expected total number of iterations."""

    @abstractmethod
    def get_progress(self) -> int:
        """Get the number of iterations that have been completed."""

    @abstractmethod
    def set_progress(self, pos: int):
        """Set the number of iterations that have been completed."""

    @abstractmethod
    def update(self, n: int):
        """Increment the number of iterations that have been completed."""

    @abstractmethod
    def close(self):
        """Close the progress bar."""


class TqdmProgressBar(JavaIProgressBar):
    def __init__(self, progress_bar: tqdm):
        super().__init__()
        self.progress_bar = progress_bar

    def get_total(self) -> int:
        return self.progress_bar.total

    def set_total(self, total: int):
        self.progress_bar.total = total

    def get_progress(self) -> int:
        return self.progress_bar.n

    def set_progress(self, pos: int):
        self.update(pos - self.get_progress())

    def update(self, n: int) -> None:
        self.progress_bar.update(n)

    def close(self) -> None:
        self.progress_bar.close()


def resolve_progress_bar(
    progress_bar: Union[bool, tqdm], description: str
) -> Union[JavaIProgressBar, None]:
    """Helper function to turn a ``ProgressBarArg`` type into a ``JavaIProgressBar``."""
    if isinstance(progress_bar, bool) and progress_bar is True:
        return TqdmProgressBar(tqdm(desc=description))
    elif progress_bar is False:
        return None
    elif isinstance(progress_bar, tqdm):
        return TqdmProgressBar(progress_bar)
    else:
        raise TypeError(f"Invalid progress_bar type: {type(progress_bar)}")
