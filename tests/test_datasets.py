from typing import Sized, Type
import capymoa.datasets as capymoa_datasets
from capymoa.stream import Stream
from capymoa.datasets import ElectricityTiny
from tempfile import TemporaryDirectory
import pytest
import numpy as np
import platform
from capymoa.datasets._downloader import _DownloadableDataset
import inspect

_ALL_DOWNLOADABLE_DATASET = [
    cls
    for _, cls in inspect.getmembers(capymoa_datasets)
    if inspect.isclass(cls) and issubclass(cls, _DownloadableDataset)
]
"""Automatically collect all datasets that are instances of DownloadableDataset
from the capymoa_datasets module.
"""


def test_electricity_tiny_auto_download():
    # If windows skip
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
    assert schema.is_regression() is False
    assert schema.is_classification() is True

    for y_index, y_value in enumerate(schema.get_label_values()):
        assert schema.get_index_for_label(y_value) == y_index
        assert schema.get_value_for_index(y_index) == y_value


@pytest.mark.skip("This test is too slow")
@pytest.mark.parametrize("dataset_type", _ALL_DOWNLOADABLE_DATASET)
def test_all_datasets(dataset_type: Type[_DownloadableDataset]):
    with TemporaryDirectory() as tmp_dir:
        dataset_arff = dataset_type(directory=tmp_dir)
        assert isinstance(dataset_arff, Stream)

        i = 0
        while dataset_arff.has_more_instances():
            dataset_arff.next_instance()
            i += 1

        assert str(dataset_arff)
        assert isinstance(dataset_arff, Sized), "Dataset must be an instance of Sized"
        assert len(dataset_arff) == i, "Dataset length must be correct"
        dataset_arff.restart()

        try:
            dataset_csv = dataset_type(directory=tmp_dir, file_type="csv")
            assert isinstance(dataset_csv, Stream)
        except ValueError:
            return  # If the dataset does not support CSV, skip the rest of the test

        # Both should return a schema object
        assert dataset_arff.get_schema() is not None
        assert dataset_csv.get_schema() is not None

        i = 0
        while dataset_arff.has_more_instances() and dataset_csv.has_more_instances():
            instance_arff = dataset_arff.next_instance()
            instance_csv = dataset_csv.next_instance()

            assert instance_arff.x == pytest.approx(instance_csv.x)
            if dataset_csv.get_schema().is_classification():
                assert instance_arff.y_index == pytest.approx(instance_csv.y_index)
            elif dataset_csv.get_schema().is_regression():
                assert instance_arff.y_value == pytest.approx(instance_csv.y_value)

            i += 1

        # Both datasets should be exhausted by now.
        assert not dataset_arff.has_more_instances()
        assert not dataset_csv.has_more_instances()

        # The datasets should be restartable.
        dataset_arff.restart()
        dataset_csv.restart()

        # After restarting, the datasets should have more instances.
        assert dataset_arff.has_more_instances()
        assert dataset_csv.has_more_instances()

        # The string representation of the datasets should not throw an error
        assert str(dataset_arff)
        assert str(dataset_csv)
        # The datasets should be the same length, and should have a size.
        assert isinstance(dataset_arff, Sized)
        assert isinstance(dataset_csv, Sized)
        assert len(dataset_arff) == len(dataset_csv) == i
