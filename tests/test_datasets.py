from capymoa.datasets import *
from tempfile import TemporaryDirectory
import pytest
import numpy as np


def test_electricity_tiny():
    with TemporaryDirectory() as tmp_dir:
        # Ensure that the dataset is not downloaded
        with pytest.raises(FileNotFoundError):
            stream = ElectricityTiny(directory=tmp_dir, auto_download=False)

        stream = ElectricityTiny(directory=tmp_dir)
        first_instance: np.ndarray = stream.next_instance().x()

        assert first_instance == pytest.approx(
            np.array([0, 0.056443, 0.439155, 0.003467, 0.422915, 0.414912])
        )

        # This should still work because the dataset is downloaded
        stream = ElectricityTiny(directory=tmp_dir, auto_download=False)


@pytest.mark.skip(reason="Too slow for CI")
def test_all_datasets():
    for dataset in [Hyper100k, CovtFD, Covtype, RBFm_100k, RTG_2abrupt]:
        with TemporaryDirectory() as tmp_dir:
            stream = dataset(directory=tmp_dir)
