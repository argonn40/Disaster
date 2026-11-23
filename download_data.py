"""Download and organise the AIDERv2 dataset from Zenodo.

AIDERv2 (KIOS CoE, University of Cyprus) contains 16,723 aerial/satellite
images in exactly four classes: Earthquake, Flood, Wildfire and Normal.
Licence: CC-BY-4.0 -- free for academic use, please cite the authors.

    python download_data.py                 # all three splits (~1.5 GB)
    python download_data.py --splits Val Test   # quick start (~300 MB)

After extraction the layout is::

    data/aider/Train/<ClassName>/*.jpg
    data/aider/Val/<ClassName>/*.jpg
    data/aider/Test/<ClassName>/*.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

RECORD_ID = "10891054"  # https://zenodo.org/records/10891054
BASE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files"

# name -> (zenodo file name, expected size in bytes, md5)
SPLIT_FILES: Dict[str, Dict[str, object]] = {
    "Train": {
        "file": "Train.zip",
        "size": 1_195_553_208,
        "md5": "25d0357a1d3c378fbbeb1a9c543d5a45",
    },
    "Val": {
        "file": "Val.zip",
        "size": 149_411_561,
        "md5": "8fb4daca73b8ffb9c745a4dea721d08f",
    },
    "Test": {
        "file": "Test.zip",
        "size": 148_359_524,
        "md5": "30d338e94dd0fbcc55ca67315981a900",
    },
}

# Folder names inside the AIDERv2 archives.
CLASSES = ("Earthquake", "Flood", "Fire", "Normal")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def download_with_progress(url: str, destination: Path) -> None:
    """Stream a file to disk, printing a progress bar."""
    request = urllib.request.Request(url, headers={"User-Agent": "aider-download/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        started = time.time()
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    speed = downloaded / max(1e-6, time.time() - started)
                    sys.stdout.write(
                        f"\r  {percent:5.1f}%  {_human(downloaded)} / {_human(total)}"
                        f"  ({_human(speed)}/s)   "
                    )
                    sys.stdout.flush()
    sys.stdout.write("\n")


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_images(folder: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in folder.rglob("*")
    )


def _find_split_source(root: Path, split: str) -> Path | None:
    """Find the extracted folder that holds the class sub-folders."""
    best: Path | None = None
    for dirpath, _dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        if current.name.lower() != split.lower():
            continue
        has_class_dir = any(
            child.is_dir() and _contains_images(child) for child in current.iterdir()
        )
        if has_class_dir and (best is None or len(current.parts) < len(best.parts)):
            best = current
    return best


def _find_class_container(root: Path) -> Path | None:
    """Archive root that already holds the class folders (AIDERv2 zips)."""
    best: Path | None = None
    for dirpath, _dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        class_dirs = [child for child in current.iterdir() if child.is_dir()]
        if len(class_dirs) >= 2 and all(_contains_images(child) for child in class_dirs):
            if best is None or len(current.parts) < len(best.parts):
                best = current
    return best


def merge_split(source: Path, target: Path) -> int:
    """Move the class folders of ``source`` into ``target``; returns #images."""
    target.mkdir(parents=True, exist_ok=True)
    for class_dir in sorted(child for child in source.iterdir() if child.is_dir()):
        destination = target / class_dir.name
        if destination.exists():
            continue
        shutil.move(str(class_dir), str(destination))
    return count_images(target)


def count_images(folder: Path) -> int:
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def print_summary(dest: Path, splits: List[str]) -> None:
    print("\nDataset summary")
    print("-" * 44)
    grand_total = 0
    for split in splits:
        split_dir = dest / split
        if not split_dir.exists():
            print(f"  {split:<6} (missing)")
            continue
        counts = {
            class_dir.name: count_images(class_dir)
            for class_dir in sorted(split_dir.iterdir())
            if class_dir.is_dir()
        }
        total = sum(counts.values())
        grand_total += total
        detail = ", ".join(f"{name}: {value}" for name, value in counts.items())
        print(f"  {split:<6} {total:>6} images   [{detail}]")
    print("-" * 44)
    print(f"  total  {grand_total:>6} images")
    if grand_total:
        print(f"\nReady. Next: python train.py --data-dir {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download AIDERv2 (4-class aerial disaster dataset) from Zenodo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dest", default="data/aider", help="target data folder")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Val", "Test"],
        choices=list(SPLIT_FILES),
        help="which splits to download",
    )
    parser.add_argument(
        "--keep-zips", action="store_true", help="keep the downloaded archives"
    )
    parser.add_argument(
        "--skip-md5", action="store_true", help="skip checksum verification"
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        info = SPLIT_FILES[split]
        filename = str(info["file"])
        target_split = dest / split

        if target_split.exists() and count_images(target_split) > 0:
            print(f"[{split}] already present at {target_split} - skipping download.")
            continue

        print(f"\n[{split}] downloading {filename} (~{_human(float(info['size']))})")
        url = f"{BASE_URL}/{filename}/content"
        zip_path = (dest if args.keep_zips else Path(tempfile.gettempdir())) / filename

        try:
            download_with_progress(url, zip_path)
        except Exception as error:  # noqa: BLE001 - user facing message
            print(f"\n[error] download failed: {error}")
            print("Download it manually from https://zenodo.org/records/10891054")
            print(f"and place the extracted folders in {dest.resolve()}")
            return 1

        if not args.skip_md5:
            print("  verifying md5 ...", end="", flush=True)
            digest = md5sum(zip_path)
            if digest != info["md5"]:
                print(f" MISMATCH\n  expected {info['md5']}\n  got      {digest}")
                print("  The download is corrupted; delete it and retry.")
                return 1
            print(" ok")

        print(f"  extracting to {dest} ...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(tmp_path)
            source = _find_split_source(tmp_path, split)
            if source is None:
                # The AIDERv2 zips put <Class>/ folders at the archive root.
                source = _find_class_container(tmp_path)
            if source is None:
                print("  [error] could not find the class folders inside the archive")
                return 1
            total = merge_split(source, target_split)
        print(f"  [{split}] {total} images ready.")

        if not args.keep_zips and zip_path.exists():
            zip_path.unlink()

    print_summary(dest, args.splits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
