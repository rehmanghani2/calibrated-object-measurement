
"""Analyze camera calibration quality and validate saved parameters.

Reads:
    calibration/calibration_report.json
    calibration/reprojection_errors.csv
    calibration/camera_calibration.npz

Writes:
    calibration/diagnostics/calibration_diagnostics.json
    calibration/diagnostics/calibration_diagnostics.txt

This script does not modify the original calibration results.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from statistics import mean, median

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
DIAGNOSTICS_DIR = CALIBRATION_DIR / "diagnostics"

REPORT_PATH = CALIBRATION_DIR / "calibration_report.json"
ERRORS_PATH = CALIBRATION_DIR / "reprojection_errors.csv"
PARAMETERS_PATH = CALIBRATION_DIR / "camera_calibration.npz"

LOGGER = logging.getLogger("calibration_diagnostics")


def configure_logging() -> Path:
    """Configure console and persistent diagnostic logging."""
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    log_path = DIAGNOSTICS_DIR / "calibration_diagnostics.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_path,
        mode="a",
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

    return log_path


def load_json_report() -> dict:
    """Load the calibration report."""
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"Calibration report not found: {REPORT_PATH}")

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_reprojection_errors() -> list[dict]:
    """Load per-image reprojection errors."""
    if not ERRORS_PATH.is_file():
        raise FileNotFoundError(f"Reprojection CSV not found: {ERRORS_PATH}")

    with ERRORS_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The reprojection error CSV contains no records.")

    for row in rows:
        row["reprojection_error_px"] = float(
            row["reprojection_error_px"]
        )
        row["corner_count"] = int(row["corner_count"])

    return rows


def validate_camera_parameters() -> dict:
    """Validate camera matrix and distortion coefficient arrays."""
    if not PARAMETERS_PATH.is_file():
        raise FileNotFoundError(
            f"Camera parameter file not found: {PARAMETERS_PATH}"
        )

    with np.load(PARAMETERS_PATH) as data:
        camera_matrix = data["camera_matrix"]
        distortion = data["distortion_coefficients"]

        image_width = int(data["image_width"])
        image_height = int(data["image_height"])

        matrix_shape_valid = camera_matrix.shape == (3, 3)
        distortion_shape_valid = distortion.size >= 4

        finite_values = (
            np.isfinite(camera_matrix).all()
            and np.isfinite(distortion).all()
        )

        focal_lengths_positive = (
            camera_matrix[0, 0] > 0
            and camera_matrix[1, 1] > 0
        )

        valid = (
            matrix_shape_valid
            and distortion_shape_valid
            and finite_values
            and focal_lengths_positive
            and image_width > 0
            and image_height > 0
        )

        return {
            "valid": bool(valid),
            "camera_matrix_shape": list(camera_matrix.shape),
            "distortion_coefficients_shape": list(distortion.shape),
            "image_width_px": image_width,
            "image_height_px": image_height,
            "matrix_finite": bool(np.isfinite(camera_matrix).all()),
            "distortion_finite": bool(np.isfinite(distortion).all()),
            "focal_lengths_positive": bool(focal_lengths_positive),
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": distortion.tolist(),
        }


def calculate_error_statistics(
    rows: list[dict],
) -> dict:
    """Calculate descriptive statistics for reprojection errors."""
    errors = [
        row["reprojection_error_px"]
        for row in rows
    ]

    return {
        "image_count": len(errors),
        "mean_error_px": mean(errors),
        "median_error_px": median(errors),
        "minimum_error_px": min(errors),
        "maximum_error_px": max(errors),
        "per_image_errors": rows,
        "highest_error_images": sorted(
            rows,
            key=lambda row: row["reprojection_error_px"],
            reverse=True,
        )[:10],
    }


def build_diagnostics() -> dict:
    """Build a complete, non-destructive diagnostic report."""
    report = load_json_report()
    errors = load_reprojection_errors()
    parameters = validate_camera_parameters()
    statistics = calculate_error_statistics(errors)

    detected_count = int(
        report["calibration_images_detected"]
    )
    failed_count = int(
        report["calibration_images_failed"]
    )
    total_count = int(
        report["calibration_images_found"]
    )

    diagnostics = {
        "calibration_summary": {
            "images_found": total_count,
            "images_detected": detected_count,
            "images_failed": failed_count,
            "detection_rate_percent": (
                detected_count / total_count * 100
                if total_count else 0.0
            ),
            "failed_image_names": report.get(
                "failed_image_names",
                [],
            ),
            "image_width_px": report[
                "normalized_image_width_px"
            ],
            "image_height_px": report[
                "normalized_image_height_px"
            ],
            "opencv_rms_reprojection_error_px": report[
                "opencv_rms_reprojection_error_px"
            ],
            "overall_reprojection_error_px": report[
                "overall_reprojection_error_px"
            ],
        },
        "reprojection_statistics": statistics,
        "camera_parameter_validation": parameters,
        "assessment_checks": {
            "minimum_20_detected_images_met": detected_count >= 20,
            "all_detected_images_have_error_records": (
                len(errors) == detected_count
            ),
            "camera_parameters_valid": parameters["valid"],
        },
        "interpretation": {
            "reprojection_error": (
                "Requires engineering review. No universal acceptance "
                "threshold is imposed by this diagnostic script."
            ),
            "calibration_status": (
                "PRELIMINARY - review image-level errors and calibration "
                "image quality before using these parameters."
            ),
        },
    }

    return diagnostics


def write_text_report(diagnostics: dict) -> Path:
    """Write a readable diagnostic summary."""
    output_path = DIAGNOSTICS_DIR / "calibration_diagnostics.txt"

    summary = diagnostics["calibration_summary"]
    statistics = diagnostics["reprojection_statistics"]
    checks = diagnostics["assessment_checks"]

    lines = [
        "CAMERA CALIBRATION DIAGNOSTICS",
       "=" * 40,
        "",
        "IMAGE DETECTION",
        f"Images found: {summary['images_found']}",
        f"Images detected: {summary['images_detected']}",
        f"Images failed: {summary['images_failed']}",
        f"Detection rate: {summary['detection_rate_percent']:.2f}%",
        "",
        "REPROJECTION ERROR",
        f"OpenCV RMS: {summary['opencv_rms_reprojection_error_px']:.6f} px",
        f"Overall RMS: {summary['overall_reprojection_error_px']:.6f} px",
        f"Mean per-image RMS: {statistics['mean_error_px']:.6f} px",
        f"Median per-image RMS: {statistics['median_error_px']:.6f} px",
        f"Minimum per-image RMS: {statistics['minimum_error_px']:.6f} px",
        f"Maximum per-image RMS: {statistics['maximum_error_px']:.6f} px",
        "",
        "ASSESSMENT CHECKS",
    ]

    for check_name, passed in checks.items():
        lines.append(f"{check_name}: {'PASS' if passed else 'FAIL'}")

    lines.extend(["", "HIGHEST PER-IMAGE ERRORS"])

    for row in statistics["highest_error_images"]:
        lines.append(
            f"{row['image']}: "
            f"{row['reprojection_error_px']:.6f} px"
        )

    lines.extend(
        [
            "",
            "FAILED DETECTION IMAGES",
            *summary["failed_image_names"],
            "",
            diagnostics["interpretation"]["calibration_status"],
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output_path


def main() -> int:
    """Run diagnostics and persist results."""
    log_path = configure_logging()

    try:
        LOGGER.info("Starting calibration diagnostics.")

        diagnostics = build_diagnostics()

        json_path = (
            DIAGNOSTICS_DIR / "calibration_diagnostics.json"
        )
        json_path.write_text(
            json.dumps(diagnostics, indent=4),
            encoding="utf-8",
        )

        text_path = write_text_report(diagnostics)

        LOGGER.info("Diagnostic JSON: %s", json_path)
        LOGGER.info("Diagnostic text: %s", text_path)

        print("\nCALIBRATION DIAGNOSTICS")
        print("=" * 40)
        print(
            "Detected images:",
            diagnostics["calibration_summary"]["images_detected"],
        )
        print(
            "Mean per-image RMS:",
            f"{diagnostics['reprojection_statistics']['mean_error_px']:.6f} px",
        )
        print(
            "Median per-image RMS:",
            f"{diagnostics['reprojection_statistics']['median_error_px']:.6f} px",
        )
        print("\nHighest-error images:")

        for row in diagnostics["reprojection_statistics"][
            "highest_error_images"
        ]:
            print(
                f"  {row['image']}: "
                f"{row['reprojection_error_px']:.6f} px"
            )

        LOGGER.info("Diagnostics completed successfully.")
        return 0

    except Exception:
        LOGGER.exception("Calibration diagnostics failed.")
        return 1

    finally:
        LOGGER.info("Diagnostic run finished.")
        print(f"\nPersistent log: {log_path}")


if __name__ == "__main__":
    raise SystemExit(main())
