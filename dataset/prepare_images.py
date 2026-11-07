
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "dataset" / "originals"
SPLIT_DIR = ROOT / "dataset" / "splits"
OUTPUT_DIR = ROOT / "dataset" / "undistorted"
CALIBRATION_FILE = ROOT / "calibration" / "camera_calibration.npz"

EXPECTED_IMAGE_COUNT = 129
CALIBRATION_RMS_PX = 4.388319547976449

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def read_split_filenames() -> list[str]:
    """Read the committed train, validation and test filename lists."""
    filenames = []

    for split_name in ("train.txt", "val.txt", "test.txt"):
        split_path = SPLIT_DIR / split_name

        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        entries = [
            line.strip().replace("\\", "/").split("/")[-1]
            for line in split_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        logger.info("%s: %d entries", split_name, len(entries))
        filenames.extend(entries)

    normalized = [name.casefold() for name in filenames]

    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate filenames found across dataset splits.")

    if len(filenames) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_IMAGE_COUNT} images, found {len(filenames)}."
        )

    return filenames


def load_calibration():
    """Load and validate the saved OpenCV calibration parameters."""
    if not CALIBRATION_FILE.is_file():
        raise FileNotFoundError(CALIBRATION_FILE)

    with np.load(CALIBRATION_FILE) as data:
        required_keys = {
            "camera_matrix",
            "distortion_coefficients",
            "image_width",
            "image_height",
        }

        missing = required_keys.difference(data.files)
        if missing:
            raise KeyError(f"Calibration file missing keys: {sorted(missing)}")

        camera_matrix = data["camera_matrix"]
        distortion = data["distortion_coefficients"]
        width = int(data["image_width"])
        height = int(data["image_height"])

    if camera_matrix.shape != (3, 3):
        raise ValueError("Camera matrix must have shape (3, 3).")

    if not np.isfinite(camera_matrix).all():
        raise ValueError("Camera matrix contains invalid values.")

    if not np.isfinite(distortion).all():
        raise ValueError("Distortion coefficients contain invalid values.")

    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive.")

    return camera_matrix, distortion, (width, height)


def main() -> None:
    camera_matrix, distortion, calibration_size = load_calibration()
    filenames = read_split_filenames()

    logger.info("Calibration image size: %s", calibration_size)
    logger.warning(
        "Calibration is provisional. RMS reprojection error: %.4f px",
        CALIBRATION_RMS_PX,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    failures = []

    for index, filename in enumerate(filenames, start=1):
        source_path = IMAGE_DIR / filename
        output_path = OUTPUT_DIR / filename

        if not source_path.is_file():
            failures.append(f"{filename}: source image not found")
            continue

        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)

        if image is None:
            failures.append(f"{filename}: image could not be decoded")
            continue

        height, width = image.shape[:2]

# Match the orientation normalization used during calibration.
        if (width, height) == (calibration_size[1], calibration_size[0]):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            height, width = image.shape[:2]

        if (width, height) != calibration_size:
            failures.append(
                f"{filename}: normalized image size {(width, height)} "
                f"does not match calibration {calibration_size}"
            )
            continue

        # Undistort using the original camera matrix.
        # This preserves the original dimensions and field of view.
        undistorted = cv2.undistort(
            image,
            camera_matrix,
            distortion,
            None,
            camera_matrix,
        )

        if undistorted.shape != image.shape:
            failures.append(f"{filename}: output dimensions changed")
            continue

        if not cv2.imwrite(str(output_path), undistorted):
            failures.append(f"{filename}: failed to save output")
            continue

        manifest_rows.append({
            "filename": filename,
            "source": f"dataset/originals/{filename}",
            "output": f"dataset/undistorted/{filename}",
            "width_px": width,
            "height_px": height,
            "status": "success",
        })

        if index % 10 == 0 or index == len(filenames):
            logger.info("Processed %d/%d images", index, len(filenames))

    # Save a CSV manifest.
    manifest_path = OUTPUT_DIR / "image_manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "source",
                "output",
                "width_px",
                "height_px",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Save processing metadata.
    metadata = {
        "method": "cv2.undistort",
        "input_directory": "dataset/originals",
        "output_directory": "dataset/undistorted",
        "expected_images": len(filenames),
        "successfully_processed": len(manifest_rows),
        "calibration_file": "calibration/camera_calibration.npz",
        "calibration_image_size_px": list(calibration_size),
        "calibration_rms_reprojection_error_px": CALIBRATION_RMS_PX,
        "calibration_status": "provisional",
        "camera_matrix_used_as_new_camera_matrix": True,
        "output_dimensions_preserved": True,
        "failures": failures,
        "limitations": [
            "The calibration reprojection error is high.",
            "Matching image dimensions do not prove identical camera settings.",
            "Focus, zoom and lens consistency have not been independently verified.",
        ],
    }

    metadata_path = OUTPUT_DIR / "preparation_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Successfully processed: %d/%d",
        len(manifest_rows),
        len(filenames),
    )
    logger.info("Manifest: %s", manifest_path)
    logger.info("Metadata: %s", metadata_path)

    if failures:
        for failure in failures:
            logger.error("%s", failure)

        raise RuntimeError(
            f"Undistortion incomplete: {len(failures)} image(s) failed."
        )


if __name__ == "__main__":
    main()
