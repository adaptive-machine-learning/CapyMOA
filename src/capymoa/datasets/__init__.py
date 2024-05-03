"""CapyMOA comes with some datasets 'out of the box'. Simply import the dataset
and start using it, the data will be downloaded automatically if it is not
already present in the download directory. You can configure where the datasets
are downloaded to by setting an environment variable (See :mod:`capymoa.env`)

>>> from capymoa.datasets import ElectricityTiny
>>> stream = ElectricityTiny()
>>> stream.next_instance().x
array([0.      , 0.056443, 0.439155, 0.003467, 0.422915, 0.414912])

Alternatively, you may download the datasets all at once with the command line interface
provided by ``capymoa.datasets``:

..  code-block:: bash

    python -m capymoa.datasets --help

"""

from ._datasets import (
    Sensor,
    RTG_2abrupt,
    RBFm_100k,
    Hyper100k,
    Fried,
    ElectricityTiny,
    Electricity,
    CovtypeTiny,
    CovtypeNorm,
    Covtype,
    CovtFD,
)
from .downloader import get_download_dir

__all__ = [
    "Sensor",
    "RTG_2abrupt",
    "RBFm_100k",
    "Hyper100k",
    "Fried",
    "ElectricityTiny",
    "Electricity",
    "CovtypeTiny",
    "CovtypeNorm",
    "Covtype",
    "CovtFD",
    "get_download_dir",
]
