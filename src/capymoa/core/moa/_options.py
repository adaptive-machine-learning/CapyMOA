"""Internal function for setting MOA options by name."""

from collections.abc import Mapping
from typing import Any

from com.github.javacliparser import AbstractClassOption as _JclAbstractClassOption
from com.github.javacliparser import FlagOption as _FlagOption
from moa.options import AbstractClassOption as _MoaAbstractClassOption


def set_moa_options(
    moa_object: Any,
    cli: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> None:
    """Set the options of a MOA object from a CLI string or a dict.

    Pass either ``cli`` or ``options``, not both. Keys in ``options`` are MOA option
    names, for example ``"gracePeriod"``.

    >>> from moa.classifiers.trees import HoeffdingTree
    >>> ht = HoeffdingTree()
    >>> set_moa_options(ht, options={"gracePeriod": 50, "binarySplits": True})
    >>> str(ht.getOptions().getAsCLIString())
    '-g 50 -b'

    :param moa_object: The MOA object whose options are set.
    :param cli: A CLI string to set the options with.
    :param options: A dict that maps MOA option names to values.
    """
    if cli is not None and options is not None:
        raise ValueError("Pass either CLI or options, not both.")
    if cli is not None:
        moa_object.getOptions().setViaCLIString(cli)
    if options is not None:
        all_options = list(moa_object.getOptions().getOptionArray())
        for key, value in options.items():
            option = next((o for o in all_options if str(o.getName()) == key), None)
            if option is None:
                valid = sorted(str(o.getName()) for o in all_options)
                raise ValueError(f"Unknown MOA option {key!r}. Valid names: {valid}")
            _set_option_value(option, value)


def _set_option_value(option: Any, value: Any) -> None:
    if isinstance(option, (_MoaAbstractClassOption, _JclAbstractClassOption)):
        if isinstance(value, str):
            option.setValueViaCLIString(value)
        elif hasattr(value, "moa_learner"):
            option.setCurrentObject(value.moa_learner)
        elif hasattr(value, "moa_detector"):
            option.setCurrentObject(value.moa_detector)
        else:
            option.setCurrentObject(value)
    elif isinstance(option, _FlagOption):
        option.setValue(bool(value))
    else:
        option.setValueViaCLIString(str(value))
