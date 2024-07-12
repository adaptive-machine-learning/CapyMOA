"""Module containing split criteria for decision trees.

Decision trees are built by splitting the data into groups based on a split
criterion. The split criterion is a function that measures the quality of a
split.
"""

from typing import Optional, Union
import moa.classifiers.core.splitcriteria as moa_split


class SplitCriterion:
    """Split criteria are used to evaluate the quality of a split in a decision tree."""

    _java_object: Optional[moa_split.SplitCriterion] = None

    def java_object(self) -> moa_split.SplitCriterion:
        """Return the Java object that this class wraps."""
        if self._java_object is None:
            raise RuntimeError("No Java object has been created.")
        return self._java_object


class VarianceReductionSplitCriterion(SplitCriterion):
    """Goodness of split criterion based on variance reduction."""

    def __init__(self):
        self._java_object = moa_split.VarianceReductionSplitCriterion()


class InfoGainSplitCriterion(SplitCriterion):
    """Goodness of split using information gain."""

    def __init__(self, min_branch_frac: float = 0.01):
        """
        Construct InfoGainSplitCriterion.

        :param min_branch_frac: Minimum fraction of weight required down at least two branches.
        """
        cli = []
        cli.append(f"-f {min_branch_frac}")

        self._java_object = moa_split.InfoGainSplitCriterion()
        self._java_object.getOptions().setViaCLIString(" ".join(cli))


class GiniSplitCriterion(SplitCriterion):
    """Goodness of split using Gini impurity."""

    def __init__(self):
        self._java_object = moa_split.GiniSplitCriterion()


def _split_criterion_to_cli_str(split_criterion: Union[str, SplitCriterion]) -> str:
    """Convert a split criterion to a CLI string.

    Also strips any parentheses or whitespace from the beginning and end of the string.

    >>> _split_criterion_to_cli_str("(InfoGainSplitCriterion -f 0.5)")
    'InfoGainSplitCriterion -f 0.5'
    >>> _split_criterion_to_cli_str(InfoGainSplitCriterion(0.5))
    'InfoGainSplitCriterion -f 0.5'

    :param split_criterion: The split criterion to convert
    :return: A CLI string representing the split criterion
    """
    if isinstance(split_criterion, SplitCriterion):
        java_object = split_criterion.java_object()
        cli_options = java_object.getOptions().getAsCLIString()
        return f"{java_object.getClass().getSimpleName()} {cli_options}"
    elif isinstance(split_criterion, str):
        return split_criterion.strip().strip("() ")
    else:
        raise TypeError(
            f"Expected a string or SplitCriterion, got {type(split_criterion)}"
        )
