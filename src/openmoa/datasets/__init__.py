# SPDX-License-Identifier: BSD-3-Clause
"""OpenMOA comes with some datasets 'out of the box'. Simply import the dataset
and start using it, the data will be downloaded automatically if it is not
already present in the download directory. You can configure where the datasets
are downloaded to by setting an environment variable (See :mod:`openmoa.env`)

>>> from openmoa.datasets import ElectricityTiny
>>> stream = ElectricityTiny()
>>> stream.next_instance().x
array([0.      , 0.056443, 0.439155, 0.003467, 0.422915, 0.414912])

Alternatively, you may download the datasets all at once with the command line interface
provided by ``openmoa.datasets``:

..  code-block:: bash

    python -m openmoa.datasets --help

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
    
    # Binary Classification Benchmarks
    RCV1,
    W8a,
    Adult,      # a8a
    InternetAds,
    Magic04,
    Spambase,
    Musk,
    SVMGuide3,
    German,
    Australian,
    Ionosphere,
    
    # Multi-Class Classification Benchmarks
    DryBean,
    Optdigits,
    Frogs,
    Wine,
    Splice,

    # COVID and Seagate Datasets
    SeagateBinary,
    SeagateMulti,
    C6StayHomeRequirements,
    C7InternalMovement,
)
from ._utils import get_download_dir
from . import downloader

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
    "downloader",
    "get_download_dir",
    "RCV1",
    "W8a",
    "Adult",
    "InternetAds",
    "Magic04",
    "Spambase",
    "Musk",
    "SVMGuide3",
    "German",
    "Australian",
    "Ionosphere",
    "DryBean",
    "Optdigits",
    "Frogs",
    "Wine",
    "downloader",
    "get_download_dir",
    "Splice",
    "SeagateBinary",
    "SeagateMulti",
    "C6StayHomeRequirements",
    "C7InternalMovement",
]
