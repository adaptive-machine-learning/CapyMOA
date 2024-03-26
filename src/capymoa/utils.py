def build_cli_str_from_mapping_and_locals(mapping, lcs):
    """
    A handy function for creating wrappers.
    Builds a CLI string based on a provided mapping and the current scope's local variables.

    Parameters
    ----------

    mapping
        A dict[str,str] which maps from the parameter as specified in the signature to the respective cli character. E.g., 'm = {"grace_period": "-g"}'
    lcs
        A dictionary containing the current scope's local variables. Can be obtained by calling 'locals()' inside the initializer
    """
    config_str = ""
    for key in mapping:
        set_value = lcs[key]
        is_bool = type(set_value) == bool
        if is_bool:
            str_extension = mapping[key] + " " if set_value else ""
        else:
            str_extension = f"{mapping[key]} {set_value} "
        config_str += str_extension
    return config_str
