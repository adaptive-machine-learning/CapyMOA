import wget
from pathlib import Path

if __name__ == "__main__":
    moa_path = Path("src/capymoa/jar/moa.jar")
    if not moa_path.exists():
        url = "https://homepages.ecs.vuw.ac.nz/~antonlee/capymoa/moa.jar"
        print(f"Downloading moa.jar from : {url}")
        wget.download(url, out=moa_path.resolve().as_posix())
    else:
        print(f"moa.jar already exists at {moa_path}")
