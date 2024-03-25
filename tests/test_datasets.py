from capymoa.datasets import (
    Hyper100k,
    CovtFD,
    Covtype,
    RBFm_100k,
    RTG_2abrupt,
    ElectricityTiny,
)
from tempfile import TemporaryDirectory
import pytest
import numpy as np
import platform


def test_electricity_tiny_auto_download():
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
        first_instance: np.ndarray = stream.next_instance().x

        assert first_instance == pytest.approx(
            np.array([0, 0.056443, 0.439155, 0.003467, 0.422915, 0.414912])
        )

        # This should still work because the dataset is downloaded
        stream = ElectricityTiny(directory=tmp_dir, auto_download=False)


def test_electricity_tiny_schema():
    schema = ElectricityTiny().schema
    assert schema.get_label_values() == ["0", "1"]
    assert schema.get_label_indexes() == [0, 1]
    assert schema.get_num_attributes() == 6
    assert schema.get_num_classes() == 2
    assert schema.is_regression() == False
    assert schema.is_classification() == True

    for y_index, y_value in enumerate(schema.get_label_values()):
        assert schema.get_index_for_label(y_value) == y_index
        assert schema.get_value_for_index(y_index) == y_value


@pytest.mark.skip(reason="Too slow for CI")
def test_all_datasets():
    for dataset in [Hyper100k, CovtFD, Covtype, RBFm_100k, RTG_2abrupt]:
        with TemporaryDirectory() as tmp_dir:
            dataset(directory=tmp_dir)
