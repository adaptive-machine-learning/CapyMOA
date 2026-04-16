import itertools
from capymoa import classifier as capymoa_classifier_module

"""
Limit the number of automatically generated values per numeric range.
This keeps the total number of model configurations manageable.
A larger number of samples can quickly explode the number of model configurations.
Users who need more fine-grained control can specify explicit values instead of a range.
"""
DEFAULT_RANGE_SAMPLES = 5


def generate_parameter_combinations(parameters):
    """Generate all parameter combinations based on the parameter configuration list."""

    param_space = []

    for param in parameters:
        name = param.get("parameter")
        ptype = param.get("type")
        prange = param.get("range", [])

        # Range-based parameters
        if prange and len(prange) == 2:
            min_val, max_val = prange

            if ptype == "integer":
                num_values = min(DEFAULT_RANGE_SAMPLES, max_val - min_val + 1)
                step = max(1, (max_val - min_val) // (num_values - 1))
                values = list(range(min_val, max_val + 1, step))

            elif ptype == "float":
                num_values = DEFAULT_RANGE_SAMPLES
                values = [
                    min_val + (max_val - min_val) * i / (num_values - 1)
                    for i in range(num_values)
                ]

            else:
                values = [param.get("value")]

        else:
            # Single fixed value
            values = [param.get("value")]

        param_configs = [{"parameter": name, "value": val} for val in values]
        param_space.append(param_configs)

    # Cartesian product of all parameter lists
    return list(itertools.product(*param_space))


def create_capymoa_classifier(algorithm_name, params, schema):
    """Create a CapyMOA classifier instance with the specified parameters."""
    # Map MOA class names to CapyMOA classifier classes
    capymoa_classifiers = {
        name: getattr(capymoa_classifier_module, name)
        for name in dir(capymoa_classifier_module)
    }

    # Find matching classifier in CapyMOA
    if algorithm_name in capymoa_classifiers:
        clf_class = capymoa_classifiers[algorithm_name]
    else:
        # Try to find a matching classifier by the last part of the name
        classifier_name = algorithm_name.split(".")[-1]
        matching_class = None

        # Search for matching class by name
        for moa_name, capymoa_class in capymoa_classifiers.items():
            if moa_name.endswith(classifier_name):
                matching_class = capymoa_class
                break

        if matching_class is None:
            print(f"Warning: No matching CapyMOA classifier found for {algorithm_name}")
            return None

        clf_class = matching_class

    # Build constructor kwargs
    init_kwargs = {p["parameter"]: p["value"] for p in params}

    try:
        classifier = clf_class(schema=schema, **init_kwargs)
    except TypeError as e:
        print(f"Warning: Failed constructor parameters for {clf_class.__name__}: {e}")
        classifier = clf_class(schema=schema)

    return classifier
