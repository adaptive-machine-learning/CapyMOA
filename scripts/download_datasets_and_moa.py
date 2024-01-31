import os.path
import wget
import zipfile
import shutil

from pathlib import Path

z_files = {
    "covtFD": [
        "https://homepages.ecs.vuw.ac.nz/~antonlee/capymoa/covtFD.zip",
        "zip",
    ],
    "covtype": [
        "https://homepages.ecs.vuw.ac.nz/~antonlee/capymoa/covtype.zip",
        "zip",
    ],
    "Hyper100k": [
        "https://homepages.ecs.vuw.ac.nz/~antonlee/capymoa/Hyper100k.zip",
        "zip",
    ],
    "RBFm_100k": [
        "https://homepages.ecs.vuw.ac.nz/~antonlee/capymoa/RBFm_100k.zip",
        "zip",
    ],
    "RTG_2abrupt": [
        "https://homepages.ecs.vuw.ac.nz/~antonlee/capymoa/RTG_2abrupt.zip",
        "zip",
    ],
    "sensor": [
        "https://homepages.ecs.vuw.ac.nz/~antonlee/capymoa/sensor.zip",
        "zip",
    ],
    "moa": [
        "https://homepages.ecs.vuw.ac.nz/~antonlee/capymoa/moa.jar",
        "jar",
    ],
}

for name, file_info in z_files.items():
    out_file_name = name + "." + file_info[1]
    url = file_info[0]
    print(f"Downloading {out_file_name} from : {url}")
    wget.download(url, out_file_name)
    print(f"downloaded file name: {out_file_name}")
    if file_info[1].__eq__("zip"):
        with zipfile.ZipFile(out_file_name, "r") as archive:
            for archive_file in archive.namelist():
                if archive_file.find("/" + name + ".csv") or archive_file.find(
                    "/" + name + ".arff"
                ):
                    print(f"unzip: {archive_file} to ./data/")
                    archive.extract(archive_file, "./data/")
        print(f"remove downloaded zip file: {out_file_name}")
        os.remove(out_file_name)
    elif file_info[1].__eq__("jar"):
        save_filename = Path("src/capymoa/jar") / out_file_name
        os.rename(out_file_name, save_filename.resolve())

shutil.rmtree("./data/__MACOSX")
