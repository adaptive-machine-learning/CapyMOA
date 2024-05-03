"""This module defines the command line interface for downloading datasets."""

import click
from ._source_list import SOURCE_LIST
from ._utils import get_download_dir, is_already_downloaded, download_extract
from typing import Set
from typing import Optional


@click.command()
@click.option(
    "--dataset",
    "-d",
    multiple=True,
    help="The dataset to download. If not specified, all datasets will be downloaded.",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Where should the datasets be downloaded to?"
    + " Defaults to the environment variable CAPYMOA_DATASETS_DIR or `./data` if not set.",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["arff", "csv"]),
    default="arff",
    help="The format to download. Defaults to ARFF.",
)
@click.option(
    "--force",
    "-F",
    is_flag=True,
    help="Force download even if the file exists. Defaults to False.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip the confirmation prompt.",
)
def download_datasets(
    out: Optional[str], dataset: Set[str], format: str, force: bool, yes: bool
):
    """Download one or more datasets.

    Example: ``python -m capymoa.datasets -d Sensor -d Hyper100k``

    An alternative to downloading datasets with this CLI tool is to use the
    dataset classes in ``capymoa.datasets``.
    """
    if len(dataset) != 0:
        filtered_sources = dict(filter(lambda x: x[0] in dataset, SOURCE_LIST.items()))
    else:
        filtered_sources = SOURCE_LIST

    download_dir_path = get_download_dir(out)

    # List the datasets to be downloaded
    click.echo(
        f"Downloading the following datasets to {click.format_filename(download_dir_path)}"
    )
    for name, source in filtered_sources.items():
        col_name = f"  {name:20}"
        url = getattr(source, format, None)
        if url is None:
            col_status = f"Skipped: No {format} available"
            fg = "yellow"
        elif is_already_downloaded(url, download_dir_path) and not force:
            col_status = "Skipped: Already downloaded"
            fg = "green"
        else:
            col_status = url
            fg = "blue"

        click.secho(f"{col_name} {col_status}", fg=fg)

    # Are they sure?
    if not yes:
        click.confirm("Do you want to continue?", abort=True)
        click.echo("")

    # Download the datasets
    for name, source in filtered_sources.items():
        url = getattr(source, format, None)
        if url is None:
            click.secho(f"Skipping {name}: No {format} available", fg="yellow")
            continue
        if is_already_downloaded(url, download_dir_path) and not force:
            click.secho(f"Skipping {name}: Already downloaded", fg="green")
            continue

        click.echo(f"Downloading and extracting {name} from {url}")
        extracted_filename = download_extract(url, download_dir_path)
        click.secho(f"Downloaded {name} to {extracted_filename}", fg="green")


if __name__ == "__main__":
    download_datasets()
