from ._datasets import (
    CovtFD,
    Covtype,
    RBFm_100k,
    RTG_2abrupt,
    Hyper100k,
    Sensor,
    ElectricityTiny,
    Fried,
    CovtypeTiny
)
from .downloader import get_download_dir

__all__ = [
    "Hyper100k",
    "CovtFD",
    "Covtype",
    "RBFm_100k",
    "RTG_2abrupt",
    "Sensor",
    "ElectricityTiny",
    "Fried",
    "CovtypeTiny",
    "get_download_dir",
]
