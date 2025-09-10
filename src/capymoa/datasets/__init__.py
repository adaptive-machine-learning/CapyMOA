"""CapyMOA comes with some datasets 'out of the box'. Simply import the dataset
and start using it, the data will be downloaded automatically if it is not
already present in the download directory. You can configure where the datasets
are downloaded to by setting an environment variable (See :mod:`capymoa.env`)

>>> from capymoa.datasets import ElectricityTiny
>>> stream = ElectricityTiny()
>>> stream.next_instance().x
array([0.      , 0.056443, 0.439155, 0.003467, 0.422915, 0.414912])

"""

from ._datasets import (
    Bike,
    CovtFD,
    Covtype,
    CovtypeNorm,
    CovtypeTiny,
    Electricity,
    ElectricityTiny,
    Fried,
    FriedTiny,
    Hyper100k,
    RBFm_100k,
    RTG_2abrupt,
    Sensor,
)
from ._utils import get_download_dir, download_unpacked

__all__ = [
    "Bike",
    "CovtFD",
    "Covtype",
    "CovtypeNorm",
    "CovtypeTiny",
    "Electricity",
    "ElectricityTiny",
    "Fried",
    "FriedTiny",
    "Hyper100k",
    "RBFm_100k",
    "RTG_2abrupt",
    "Sensor",
    "get_download_dir",
    "download_unpacked",
]
