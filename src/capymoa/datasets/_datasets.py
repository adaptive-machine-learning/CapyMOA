from capymoa.datasets.downloader import DownloadARFFGzip
from ._source_list import SOURCE_LIST
from ._util import identify_compressed_hosted_file


class Sensor(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = identify_compressed_hosted_file(SOURCE_LIST["Sensor"].arff)[1]
    remote_url = SOURCE_LIST["Sensor"].arff


class Hyper100k(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = identify_compressed_hosted_file(SOURCE_LIST["Hyper100k"].arff)[1]
    remote_url = SOURCE_LIST["Hyper100k"].arff


class CovtFD(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = identify_compressed_hosted_file(SOURCE_LIST["CovtFD"].arff)[1]
    remote_url = SOURCE_LIST["CovtFD"].arff


class Covtype(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = identify_compressed_hosted_file(SOURCE_LIST["Covtype"].arff)[1]
    remote_url = SOURCE_LIST["Covtype"].arff


class RBFm_100k(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = identify_compressed_hosted_file(SOURCE_LIST["RBFm_100k"].arff)[1]
    remote_url = SOURCE_LIST["RBFm_100k"].arff


class RTG_2abrupt(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = identify_compressed_hosted_file(SOURCE_LIST["RTG_2abrupt"].arff)[1]
    remote_url = SOURCE_LIST["RTG_2abrupt"].arff


class ElectricityTiny(DownloadARFFGzip):
    """A tiny version of the Electricity dataset."""

    filename = identify_compressed_hosted_file(SOURCE_LIST["ElectricityTiny"].arff)[1]
    remote_url = SOURCE_LIST["ElectricityTiny"].arff


class CovtypeTiny(DownloadARFFGzip):
    """A truncated version of the Covtype dataset with 1000 instances."""

    filename = identify_compressed_hosted_file(SOURCE_LIST["CovtypeTiny"].arff)[1]
    remote_url = SOURCE_LIST["CovtypeTiny"].arff


class Fried(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = identify_compressed_hosted_file(SOURCE_LIST["Fried"].arff)[1]
    remote_url = SOURCE_LIST["Fried"].arff
