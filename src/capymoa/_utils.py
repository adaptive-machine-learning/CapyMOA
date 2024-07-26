"""Contains private utility functions used throughout the library."""

from typing import Dict, Any
import re

# Create a single mapping dictionary
_metrics_name_mapping = {
    # general
    'instances': 'classified instances',

    # accuracy
    'accuracy': 'classifications correct (percent)',
    'kappa': 'Kappa Statistic (percent)',
    'kappa_t': 'Kappa Temporal Statistic (percent)',
    'kappa_m': 'Kappa M Statistic (percent)',
    'f1_score': 'F1 Score (percent)',
    'f1_score_{N}': 'F1 Score for class {N} (percent)',
    'precision': 'Precision (percent)',
    'precision_{N}': 'Precision for class {N} (percent)',
    'recall': 'Recall (percent)',
    'recall_{N}': 'Recall for class {N} (percent)',

    # regression
    'mae': 'mean absolute error',
    'rmse': 'root mean squared error',
    'rmae': 'relative mean absolute error',
    'rrmse': 'relative root mean squared error',
    'r2': 'coefficient of determination',
    'adjusted_r2': 'adjusted coefficient of determination',

    # prediction intervals
    'coverage': 'coverage',
    'average_length': 'average length',
    'nmpiw': 'NMPIW',

    # anomaly
    'auc': 'AUC',
    's_auc': 'sAUC',
}

# Create a reverse mapping for the metrics name
_reverse_metrics_name_mapping = {v: k for k, v in _metrics_name_mapping.items()}


def _translate_metric_name(metric_name, to='capymoa'):
    # Function to handle template-based translation
    def translate_template(template_mapping, _metric_name):
        for template, value_template in template_mapping.items():
            if '{N}' in template:
                pattern = re.escape(template).replace(r'\{N\}', r'(\d+)')
                match = re.match(pattern, _metric_name)
                if match:
                    return value_template.replace('{N}', match.group(1))
        return _metric_name

    if to == 'moa':
        translation = _metrics_name_mapping.get(metric_name)
        if translation is None:
            translation = translate_template(_metrics_name_mapping, metric_name)
        return translation
    elif to == 'capymoa':
        translation = _reverse_metrics_name_mapping.get(metric_name)
        if translation is None:
            translation = translate_template(_reverse_metrics_name_mapping, metric_name)
        return translation
    else:
        raise ValueError("Invalid translation direction. Use 'moa' or 'capymoa'.")


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


def _leaf_prediction(leaf_prediction):
    """Checks the leaf_prediction option. Internal method used to check leaf_prediction parameters.

    See :py:class:`capymoa.classifier.EFDT` and :py:class:`capymoa.classifier.HoeffdingTree`

    >>> _leaf_prediction(leaf_prediction="MajorityClass")
    0

    >>> _leaf_prediction(leaf_prediction=2)
    2
    """
    if isinstance(leaf_prediction, str):
        leaf_prediction_mapping = {"MajorityClass": 0, "NaiveBayes": 1, "NaiveBayesAdaptive": 2}
        leaf_prediction = leaf_prediction_mapping.get(leaf_prediction, None)
    if not isinstance(leaf_prediction, int) or leaf_prediction < 0 or leaf_prediction > 2:
        raise ValueError("Invalid value for leaf_prediction, valid options are 'MajorityClass' or 0, "
                         "'NaiveBayes' or 1, 'NaiveBayesAdaptive' or 2,")

    return leaf_prediction
