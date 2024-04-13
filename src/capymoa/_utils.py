"""Contains private utility functions used throughout the library."""

from typing import Dict, Any


def build_cli_str_from_mapping_and_locals(mapping: Dict[str, str], lcs: Dict[str, Any]):
    """Builds a CLI string based on a provided mapping and the current scope's local variables.

    >>> max_byte_size = 33554433
    >>> grace_period = 200
    >>> mapping = {"grace_period": "-g", "max_byte_size": "-m"}
    >>> build_cli_str_from_mapping_and_locals(mapping, locals())
    '-g (200) -m (33554433) '

    :param mapping: A dict[str,str] which maps from the parameter as specified
        in the signature to the respective cli character. E.g., 'm = {"grace_period": "-g"}'
    :param lcs: A dictionary containing the current scope's local variables.
        Can be obtained by calling 'locals()' inside the initializer
    :return: A string representing the CLI configuration
    """

    config_str = ""
    for key in mapping:
        set_value = lcs[key]
        is_bool = isinstance(set_value, bool)
        if is_bool:
            str_extension = mapping[key] + " " if set_value else ""
        else:
            # The parenthesis are used to support nested classes
            str_extension = f"{mapping[key]} ({set_value}) "
        config_str += str_extension
    return config_str


def _get_moa_creation_CLI(moa_object):
    """Returns the MOA CLI string for a given MOA object.

    >>> from moa.streams import ConceptDriftStream
    ...
    >>> stream = ConceptDriftStream() 
    >>> _get_moa_creation_CLI(stream)
    'streams.ConceptDriftStream'
    """

    moa_class_id = str(moa_object.getClass().getName())
    moa_class_id_parts = moa_class_id.split(".")
    moa_stream_str = f"{moa_class_id_parts[-2]}.{moa_class_id_parts[-1]}"

    moa_cli_creation = str(moa_object.getCLICreationString(moa_object.__class__))
    CLI = moa_cli_creation.split(" ", 1)

    if len(CLI) > 1 and len(CLI[1]) > 1:
        moa_stream_str = f"({moa_stream_str} {CLI[1]})"

    return moa_stream_str
