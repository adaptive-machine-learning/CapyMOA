from capymoa.datasets.downloader import DownloadARFFGzip

ROOT_URL = "https://homepages.ecs.vuw.ac.nz/~antonlee/capymoa/standardised/"


class Sensor(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = "sensor.arff"
    remote_url = ROOT_URL


class Hyper100k(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = "Hyper100k.arff"
    remote_url = ROOT_URL


class CovtFD(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = "covtFD.arff"
    remote_url = ROOT_URL


class Covtype(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = "covtype.arff"
    remote_url = ROOT_URL


class RBFm_100k(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = "RBFm_100k.arff"
    remote_url = ROOT_URL


class RTG_2abrupt(DownloadARFFGzip):
    # TODO: Add docstring describing the dataset and link to the original source

    filename = "RTG_2abrupt.arff"
    remote_url = ROOT_URL


class ElectricityTiny(DownloadARFFGzip):
    """A tiny version of the Electricity dataset."""

    filename = "electricity_tiny.arff"
    remote_url = ROOT_URL
