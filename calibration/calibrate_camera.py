"""Calibrate the phone camera from checkerboard photographs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np


LOGGER = logging.getLogger("camera_calibration")

# The printed board has 8 x 10 squares, hence 7 x 9 inner corners.
BOARD_PATTERNS = ((7, 9), (9, 7))
SQUARE_SIZE_MM = 25.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def normalize_orientation(image: np.ndarray) -> np.ndarray:
    """Rotate portrait images clockwise into the landscape orientation."""
    height, width = image.shape[:2]

    if height > width:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    return image


def detect_checkerboard(
    gray_image: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Detect and subpixel-refine checkerboard corners."""
    detection_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    for pattern_size in BOARD_PATTERNS:
        found, corners = cv2.findChessboardCorners(
            gray_image,
            pattern_size,
            detection_flags,
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

            return refined, pattern_size

    return None


def create_object_points(
    pattern_size: tuple[int, int],
) -> np.ndarray:
    """Create checkerboard coordinates in millimetres."""
    columns, rows = pattern_size

    points = np.zeros((rows * columns, 3), dtype=np.float32)
    points[:, :2] = (
        np.mgrid[0:columns, 0:rows]
        .T.reshape(-1, 2)
        * SQUARE_SIZE_MM
    )

    return points


def calibrate(image_directory: Path, output_directory: Path) -> None:
    """Run calibration and save parameters and diagnostic reports."""
    image_paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"No calibration images found in {image_directory}"
        )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    detected_images: list[str] = []
    failed_images: list[str] = []

    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            LOGGER.warning("Could not read image: %s", image_path.name)
            failed_images.append(image_path.name)
            continue

        normalized = normalize_orientation(image)
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

        current_size = (gray.shape[1], gray.shape[0])

        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise RuntimeError(
                "Inconsistent normalized image dimensions: "
                f"{image_path.name} has {current_size}, "
                f"expected {image_size}."
            )

        detection = detect_checkerboard(gray)

        if detection is None:
            LOGGER.warning(
                "Checkerboard not detected: %s",
                image_path.name,
            )
            failed_images.append(image_path.name)
            continue

        corners, pattern_size = detection

        object_points.append(create_object_points(pattern_size))
        image_points.append(corners)
        detected_images.append(image_path.name)

        LOGGER.info(
            "Detected %s using pattern %s",
            image_path.name,
            pattern_size,
        )

    if len(detected_images) < 20:
        raise RuntimeError(
            "Calibration requires at least 20 successfully detected "
            f"images; only {len(detected_images)} were detected."
        )

    assert image_size is not None

    rms, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = (
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
    )

    per_image_errors: list[dict[str, float | str]] = []
    total_squared_error = 0.0
    total_points = 0

    for index, image_name in enumerate(detected_images):
        projected_points, _ = cv2.projectPoints(
            object_points[index],
            rotation_vectors[index],
            translation_vectors[index],
            camera_matrix,
            distortion_coefficients,
        )

        residuals = (
            image_points[index].reshape(-1, 2)
            - projected_points.reshape(-1, 2)
        )

        squared_distances = np.sum(residuals**2, axis=1)
        point_count = len(squared_distances)

        image_rms = float(
            np.sqrt(np.mean(squared_distances))
        )

        total_squared_error += float(np.sum(squared_distances))
        total_points += point_count

        per_image_errors.append(
            {
                "image": image_name,
                "corner_count": point_count,
                "reprojection_error_px": image_rms,
            }
        )

    overall_reprojection_error = float(
        np.sqrt(total_squared_error / total_points)
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    calibration_path = output_directory / "camera_calibration.npz"

    np.savez_compressed(
        calibration_path,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        image_width=image_size[0],
        image_height=image_size[1],
        square_size_mm=SQUARE_SIZE_MM,
        board_patterns=np.array(BOARD_PATTERNS),
        orientation_policy="portrait rotated clockwise to landscape",
    )

    report = {
        "calibration_images_found": len(image_paths),
        "calibration_images_detected": len(detected_images),
        "calibration_images_failed": len(failed_images),
        "detected_image_names": detected_images,
        "failed_image_names": failed_images,
        "normalized_image_width_px": image_size[0],
        "normalized_image_height_px": image_size[1],
        "checkerboard_inner_corner_patterns": BOARD_PATTERNS,
        "checkerboard_square_size_mm": SQUARE_SIZE_MM,
        "opencv_rms_reprojection_error_px": float(rms),
        "overall_reprojection_error_px": overall_reprojection_error,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion_coefficients.tolist(),
    }

    report_path = output_directory / "calibration_report.json"

    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4)

    errors_path = output_directory / "reprojection_errors.csv"

    with errors_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "corner_count",
                "reprojection_error_px",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image_errors)

    print("\nCALIBRATION COMPLETE")
    print(f"Images found: {len(image_paths)}")
    print(f"Images detected: {len(detected_images)}")
    print(f"Images failed: {len(failed_images)}")
    print(f"Normalized image size: {image_size}")
    print(f"OpenCV RMS error: {rms:.6f} px")
    print(
        "Overall reprojection error: "
        f"{overall_reprojection_error:.6f} px"
    )
    print(f"Camera parameters: {calibration_path}")
    print(f"Calibration report: {report_path}")
    print(f"Per-image errors: {errors_path}")

    if failed_images:
        print("\nImages requiring review:")
        for image_name in failed_images:
            print(f"  - {image_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a camera using checkerboard images."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("calibration/images"),
        help="Directory containing checkerboard photographs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration"),
        help="Directory for calibration outputs.",
    )

    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    calibrate(arguments.images, arguments.output)


if __name__ == "__main__":
    main()