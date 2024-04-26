"""This is a module responsible for fetching environment variables and ensuring they
have appropriate defaults.
"""

from os import environ

CAPYMOA_DATASETS_DIR = environ.get("CAPYMOA_DATASETS_DIR", "./data")
