"""This is a module responsible for storing the URLs of the datasets."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class _Source:
    arff: str
    """The URL of the ARFF file. Can be optionally compressed."""
    csv: Optional[str]
    """The URL of the CSV file. Can be optionally compressed."""


_ROOT_URL = "https://homepages.ecs.vuw.ac.nz/~antonlee/capymoa/standardised/"
SOURCE_LIST: Dict[str, _Source] = {
    "Sensor": _Source(
        f"{_ROOT_URL}sensor.arff.gz",
        f"{_ROOT_URL}sensor.csv.gz",
    ),
    "Hyper100k": _Source(
        f"{_ROOT_URL}Hyper100k.arff.gz",
        f"{_ROOT_URL}Hyper100k.csv.gz",
    ),
    "CovtFD": _Source(
        f"{_ROOT_URL}covtFD.arff.gz",
        f"{_ROOT_URL}covtFD.csv.gz",
    ),
    "Covtype": _Source(
        f"{_ROOT_URL}covtype.arff.gz",
        f"{_ROOT_URL}covtype.csv.gz",
    ),
    "RBFm_100k": _Source(
        f"{_ROOT_URL}RBFm_100k.arff.gz",
        f"{_ROOT_URL}RBFm_100k.csv.gz",
    ),
    "RTG_2abrupt": _Source(
        f"{_ROOT_URL}RTG_2abrupt.arff.gz",
        f"{_ROOT_URL}RTG_2abrupt.csv.gz",
    ),
    "ElectricityTiny": _Source(
        f"{_ROOT_URL}electricity_tiny.arff.gz",
        f"{_ROOT_URL}electricity_tiny.csv.gz",
    ),
    "CovtypeTiny": _Source(f"{_ROOT_URL}covtype_n1000.arff.gz", None),
    "Fried": _Source(
        f"{_ROOT_URL}fried.arff.gz",
        f"{_ROOT_URL}fried.csv.gz",
    ),
}
