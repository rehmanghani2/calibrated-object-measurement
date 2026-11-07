
"""Generate visual diagnostics for checkerboard calibration photographs.

Outputs:
    calibration/diagnostics/overlays/
        Full-resolution checkerboard detection overlays.

    calibration/diagnostics/checkerboard_contact_sheet.jpg
        Contact sheet for reviewing all calibration photographs.

The script does not modify calibration parameters or source images.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIRECTORY = PROJECT_ROOT / "calibration" / "images"
DIAGNOSTICS_DIRECTORY = PROJECT_ROOT / "calibration" / "diagnostics"
OVERLAY_DIRECTORY = DIAGNOSTICS_DIRECTORY / "overlays"
ERRORS_FILE = PROJECT_ROOT / "calibration" / "reprojection_errors.csv"

BOARD_PATTERNS = ((7, 9), (9, 7))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

THUMBNAIL_WIDTH = 480
THUMBNAIL_HEIGHT = 360
GRID_COLUMNS = 4

LOGGER = logging.getLogger("calibration_image_review")


def configure_logging() -> None:
    """Configure persistent and console logging."""
    DIAGNOSTICS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    log_path = DIAGNOSTICS_DIRECTORY / "image_review.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True,
    )


def normalize_orientation(image: np.ndarray) -> np.ndarray:
    """Normalize portrait photographs to landscape orientation."""
    height, width = image.shape[:2]

    if height > width:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    return image


def load_reprojection_errors() -> dict[str, float]:
    """Load previously calculated per-image reprojection errors."""
    if not ERRORS_FILE.is_file():
        raise FileNotFoundError(
            f"Reprojection error file not found: {ERRORS_FILE}"
        )

    with ERRORS_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = csv.DictReader(file)

        return {
            row["image"]: float(row["reprojection_error_px"])
            for row in rows
        }


def detect_checkerboard(
    gray_image: np.ndarray,
) -> tuple[bool, np.ndarray | None, tuple[int, int] | None]:
    """Detect checkerboard corners using the calibration settings."""
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    for pattern_size in BOARD_PATTERNS:
        found, corners = cv2.findChessboardCorners(
            gray_image,
            pattern_size,
            flags,
        )

        if not found:
            found, corners = cv2.findChessboardCornersSB(
                gray_image,
                pattern_size,
            )

        if found and corners is not None:
            refined = cv2.cornerSubPix(
                gray_image,
                corners,
                (11, 11),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS
                    | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )

            return True, refined, pattern_size

    return False, None, None


def create_thumbnail(
    image: np.ndarray,
    filename: str,
    error: float | None,
    detected: bool,
) -> np.ndarray:
    """Create a labelled thumbnail while preserving image proportions."""
    height, width = image.shape[:2]

    scale = min(
        THUMBNAIL_WIDTH / width,
        THUMBNAIL_HEIGHT / height,
    )

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH, 3),
        dtype=np.uint8,
    )

    y_offset = (THUMBNAIL_HEIGHT - resized.shape[0]) // 2
    x_offset = (THUMBNAIL_WIDTH - resized.shape[1]) // 2

    canvas[
        y_offset:y_offset + resized.shape[0],
        x_offset:x_offset + resized.shape[1],
    ] = resized

    status = "DETECTED" if detected else "NOT DETECTED"

    cv2.putText(
        canvas,
        filename,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if error is not None:
        detail = f"{status} | RMS: {error:.3f} px"
    else:
        detail = status

    cv2.putText(
        canvas,
        detail,
        (10, THUMBNAIL_HEIGHT - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0) if detected else (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def create_contact_sheet(thumbnails: list[np.ndarray]) -> np.ndarray:
    """Arrange all image thumbnails into a contact sheet."""
    rows = (
        len(thumbnails) + GRID_COLUMNS - 1
    ) // GRID_COLUMNS

    sheet = np.zeros(
        (
            rows * THUMBNAIL_HEIGHT,
            GRID_COLUMNS * THUMBNAIL_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    for index, thumbnail in enumerate(thumbnails):
        row = index // GRID_COLUMNS
        column = index % GRID_COLUMNS

        y_start = row * THUMBNAIL_HEIGHT
        x_start = column * THUMBNAIL_WIDTH

        sheet[
            y_start:y_start + THUMBNAIL_HEIGHT,
            x_start:x_start + THUMBNAIL_WIDTH,
        ] = thumbnail

    return sheet


def main() -> None:
    """Generate overlays and the checkerboard contact sheet."""
    configure_logging()
    OVERLAY_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"Calibration image directory not found: {IMAGE_DIRECTORY}"
        )

    errors = load_reprojection_errors()

    image_paths = sorted(
        path
        for path in IMAGE_DIRECTORY.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError("No calibration photographs were found.")

    thumbnails: list[np.ndarray] = []
    detected_count = 0
    failed_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            LOGGER.error("Unable to read %s", image_path.name)
            failed_count += 1
            continue

        normalized = normalize_orientation(image)
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

        found, corners, pattern_size = detect_checkerboard(gray)

        overlay = normalized.copy()
        error = errors.get(image_path.name)

        if found and corners is not None and pattern_size is not None:
            cv2.drawChessboardCorners(
                overlay,
                pattern_size,
                corners,
                True,
            )
            detected_count += 1
            LOGGER.info("Detected: %s", image_path.name)
        else:
            failed_count += 1
            LOGGER.warning("Not detected: %s", image_path.name)

        overlay_path = OVERLAY_DIRECTORY / (
            f"{image_path.stem}_overlay.jpg"
        )

        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(
                f"Failed to save overlay: {overlay_path}"
            )

        thumbnails.append(
            create_thumbnail(
                overlay,
                image_path.name,
                error,
                found,
            )
        )

    contact_sheet = create_contact_sheet(thumbnails)

    contact_sheet_path = (
        DIAGNOSTICS_DIRECTORY / "checkerboard_contact_sheet.jpg"
    )

    if not cv2.imwrite(
        str(contact_sheet_path),
        contact_sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    ):
        raise RuntimeError("Failed to save checkerboard contact sheet.")

    LOGGER.info("Images reviewed: %d", len(image_paths))
    LOGGER.info("Detections: %d", detected_count)
    LOGGER.info("Detection failures: %d", failed_count)
    LOGGER.info("Contact sheet: %s", contact_sheet_path)
    LOGGER.info("Full-resolution overlays: %s", OVERLAY_DIRECTORY)


if __name__ == "__main__":
    main()
