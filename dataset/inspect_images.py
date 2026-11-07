"""Inspect source image dimensions before camera undistortion."""

from collections import Counter
from pathlib import Path
import logging

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIRECTORY = PROJECT_ROOT / "dataset" / "originals"
SPLIT_DIRECTORY = PROJECT_ROOT / "dataset" / "splits"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def read_split_files() -> set[str]:
    """Read image filenames from all committed dataset splits."""
    filenames: set[str] = set()

    for split_name in ("train.txt", "val.txt", "test.txt"):
        split_path = SPLIT_DIRECTORY / split_name

        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        entries = {
            line.strip()
            for line in split_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        }

        LOGGER.info("%s: %d entries", split_name, len(entries))
        filenames.update(entries)

    return filenames


def main() -> None:
    """Report source image dimensions and split consistency."""
    if not IMAGE_DIRECTORY.is_dir():
        raise FileNotFoundError(IMAGE_DIRECTORY)

    images = sorted(IMAGE_DIRECTORY.glob("*.jpg"))
    dimensions: Counter[tuple[int, int]] = Counter()
    unreadable: list[str] = []

    for image_path in images:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            unreadable.append(image_path.name)
            continue

        height, width = image.shape[:2]
        dimensions[(width, height)] += 1

    split_entries = read_split_files()
    split_files = {Path(entry).name for entry in split_entries}
    actual_files = {path.name for path in images}

    LOGGER.info("Original JPEG files: %d", len(images))
    LOGGER.info("Readable images: %d", sum(dimensions.values()))

    print("\nImage dimensions (width x height):")
    for (width, height), count in sorted(dimensions.items()):
        print(f"  {width} x {height}: {count} images")

    print("\nSplit consistency:")
    print(f"  Split entries: {len(split_files)}")
    print(f"  Missing from originals: {len(split_files - actual_files)}")
    print(f"  Not listed in splits: {len(actual_files - split_files)}")

    if unreadable:
        print("\nUnreadable images:")
        for filename in unreadable:
            print(f"  {filename}")

    if len(dimensions) > 1:
        LOGGER.warning("Multiple image dimensions found; investigate before undistortion.")

    if unreadable:
        LOGGER.warning("Unreadable images must be investigated before processing.")

    if split_files != actual_files:
        LOGGER.warning("Image files and committed splits do not match.")

    LOGGER.info("Inspection complete.")


if __name__ == "__main__":
    main()