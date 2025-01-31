from typing import Sized, Type
import capymoa.datasets as capymoa_datasets
from capymoa.datasets import ElectricityTiny
from tempfile import TemporaryDirectory
import pytest
import numpy as np
import platform
from capymoa.datasets.downloader import DownloadableDataset
from capymoa.stream import stream_from_file
from subprocess import run
import inspect

_ALL_DOWNLOADABLE_DATASET = [
    cls
    for _, cls in inspect.getmembers(capymoa_datasets)
    if inspect.isclass(cls) and issubclass(cls, DownloadableDataset)
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


def test_downloader_cli():
    # If windows skip
    if platform.system() == "Windows":
        # TODO: Explicitly closing streams might help but MOA does not support
        # this yet.
        pytest.skip("Skipping on Windows, because TemporaryDirectory fails to cleanup.")

    with TemporaryDirectory() as tmp_dir:
        cmd = [
            "python",
            "-m",
            "capymoa.datasets",
            "--out",
            tmp_dir,
            "--dataset",
            "ElectricityTiny",
            "--yes",
        ]
        run([*cmd, "--format", "csv"], check=True)
        run([*cmd, "--format", "arff"], check=True)

        csv_stream = stream_from_file(tmp_dir + "/electricity_tiny.csv")
        arff_stream = stream_from_file(tmp_dir + "/electricity_tiny.arff")

        while not csv_stream.has_more_instances():
            csv_instance = csv_stream.next_instance()
            arff_instance = arff_stream.next_instance()
            assert csv_instance.x == pytest.approx(arff_instance.x)


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
def test_all_datasets(dataset_type: Type[DownloadableDataset]):
    with TemporaryDirectory() as tmp_dir:
        dataset = dataset_type(directory=tmp_dir)

        i = 0
        while dataset.has_more_instances():
            dataset.next_instance()
            i += 1

        assert str(dataset)
        assert isinstance(dataset, Sized), "Dataset must be an instance of Sized"
        assert len(dataset) == i, "Dataset length must be correct"
