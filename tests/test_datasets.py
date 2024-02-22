from capymoa.datasets import *
from tempfile import TemporaryDirectory
import pytest
import numpy as np
import platform

def test_electricity_tiny():
    # If windows and python3.9 skip
    if platform.system() == "Windows":
        # TODO: Explicitly closing streams might help but MOA does not support
        # this yet.
        pytest.skip("Skipping on Windows, because TemporaryDirectory fails to cleanup.")

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
