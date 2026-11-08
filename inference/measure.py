
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.unet import UNet
from dataset.auto_label import Config, find_card_region


# Physical dimensions of the playing-card reference.
CARD_WIDTH_MM = 56.5
CARD_HEIGHT_MM = 87.0

# The rectified card plane uses 10 pixels per millimetre.
CARD_WIDTH_PX = 565
CARD_HEIGHT_PX = 870

INPUT_WIDTH = 384
INPUT_HEIGHT = 288
PROBABILITY_THRESHOLD = 0.5


def normalize_orientation(image: np.ndarray) -> np.ndarray:
    """Match the landscape orientation used during camera calibration."""
    if image.shape[0] > image.shape[1]:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def load_calibration(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    data = np.load(path)

    required = {"camera_matrix", "distortion_coefficients"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Calibration file missing keys: {sorted(missing)}")

    camera_matrix = data["camera_matrix"]
    distortion = data["distortion_coefficients"]

    if camera_matrix.shape != (3, 3):
        raise ValueError("Camera matrix must have shape (3, 3).")
    if not np.isfinite(camera_matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("Calibration contains non-finite values.")

    return camera_matrix, distortion


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model = UNet(in_channels=3, out_channels=1, base_channels=16)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state = checkpoint.get("model_state_dict")
    if state is None:
        raise ValueError("Checkpoint has no model_state_dict.")

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_mask(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
):
    resized = cv2.resize(
        image,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(
        rgb.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probability_small = torch.sigmoid(logits)[0, 0].cpu().numpy()

    probability = cv2.resize(
        probability_small,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    binary = (probability >= PROBABILITY_THRESHOLD).astype(np.uint8) * 255

    # Keep the largest connected foreground component for this
    # single-object measurement task.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if count <= 1:
        return None, probability, 0.0

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8) * 255

    area = int(np.count_nonzero(mask))
    confidence = float(np.mean(probability[mask > 0])) if area else 0.0

    return mask, probability, confidence


def ordered_card_corners(card_mask: np.ndarray):
    contours, _ = cv2.findContours(
        card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No playing-card region detected.")

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        raise ValueError("Detected card region is too small.")

    rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect).astype(np.float32)

    # Sort the four corners cyclically around their centre.
    centre = corners.mean(axis=0)
    angles = np.arctan2(
        corners[:, 1] - centre[1],
        corners[:, 0] - centre[0],
    )
    corners = corners[np.argsort(angles)]

    edge_lengths = np.array([
        np.linalg.norm(corners[(i + 1) % 4] - corners[i])
        for i in range(4)
    ])

    # The short card edge corresponds to 56.5 mm.
    short_edge_index = int(np.argmin(edge_lengths))
    corners = np.roll(corners, -short_edge_index, axis=0)

    short_length = float(edge_lengths[short_edge_index])
    long_length = float(edge_lengths[(short_edge_index + 1) % 4])

    if short_length <= 0:
        raise ValueError("Invalid card dimensions.")

    observed_ratio = long_length / short_length
    expected_ratio = CARD_HEIGHT_MM / CARD_WIDTH_MM

    if not 1.25 <= observed_ratio <= 1.90:
        raise ValueError(
            f"Card geometry is implausible: aspect ratio {observed_ratio:.3f}"
        )

    return corners, observed_ratio


def create_card_homography(
    card_mask: np.ndarray,
    full_width: int,
    full_height: int,
):
    corners, ratio = ordered_card_corners(card_mask)

    # Card mask was detected at reduced resolution; map corners
    # back into the full-resolution undistorted image.
    work_height, work_width = card_mask.shape
    corners[:, 0] *= full_width / work_width
    corners[:, 1] *= full_height / work_height

    destination = np.array(
        [
            [0, 0],
            [CARD_WIDTH_PX - 1, 0],
            [CARD_WIDTH_PX - 1, CARD_HEIGHT_PX - 1],
            [0, CARD_HEIGHT_PX - 1],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(corners, destination)
    if not np.isfinite(homography).all():
        raise ValueError("Could not calculate a valid card homography.")

    return homography, corners, ratio


def measure_casing(mask: np.ndarray, homography: np.ndarray):
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 20:
        raise ValueError("Casing segmentation is empty or too small.")

    transformed = cv2.perspectiveTransform(
        points.astype(np.float32), homography
    )

    if not np.isfinite(transformed).all():
        raise ValueError("Casing transformation produced invalid coordinates.")

    rect = cv2.minAreaRect(transformed)
    side_a, side_b = rect[1]

    if min(side_a, side_b) <= 0:
        raise ValueError("Could not determine casing dimensions.")

    short_px = min(side_a, side_b)
    long_px = max(side_a, side_b)

    width_mm = short_px / 10.0
    height_mm = long_px / 10.0

    return width_mm, height_mm, transformed


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    card_corners: np.ndarray,
    width_mm: float,
    height_mm: float,
    confidence: float,
):
    overlay = image.copy()

    tint = overlay.copy()
    tint[mask > 0] = (0, 220, 0)
    overlay = cv2.addWeighted(tint, 0.35, overlay, 0.65, 0)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 5)

    cv2.polylines(
        overlay,
        [card_corners.astype(np.int32)],
        True,
        (255, 100, 0),
        5,
    )

    lines = [
        f"Width: {width_mm:.1f} mm",
        f"Height: {height_mm:.1f} mm",
        f"Segmentation confidence: {confidence:.3f}",
    ]

    y = 70
    for line in lines:
        cv2.putText(
            overlay,
            line,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        y += 65

    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Measure a phone casing using a playing-card reference."
    )
    parser.add_argument("--image", required=True, help="Input photograph.")
    parser.add_argument(
        "--calibration",
        default=str(ROOT / "calibration" / "camera_calibration.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "models" / "outputs" / "best_unet.pt"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "measurement" / "results"),
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    torch.set_num_threads(4)
    device = torch.device("cpu")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = normalize_orientation(image)

    camera_matrix, distortion = load_calibration(Path(args.calibration))
    undistorted = cv2.undistort(
        image, camera_matrix, distortion, None, camera_matrix
    )

    model, checkpoint = load_model(Path(args.checkpoint), device)

    mask, probability, confidence = predict_mask(
        undistorted, model, device
    )
    if mask is None:
        raise RuntimeError(
            "MEASUREMENT REFUSED: no casing foreground was detected."
        )

    # Detect the reference card using the same card-region method
    # used by the dataset annotation pipeline.
    cfg = Config()
    scale = min(1.0, cfg.work_max_side / max(undistorted.shape[:2]))
    work = cv2.resize(
        undistorted,
        (
            int(round(undistorted.shape[1] * scale)),
            int(round(undistorted.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    ) if scale < 1.0 else undistorted

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    card_mask = find_card_region(hsv, cfg)

    if not card_mask.any():
        raise RuntimeError(
            "MEASUREMENT REFUSED: the playing-card reference was not detected."
        )

    homography, card_corners, card_ratio = create_card_homography(
        card_mask, undistorted.shape[1], undistorted.shape[0]
    )

    width_mm, height_mm, _ = measure_casing(mask, homography)

    overlay = create_overlay(
        undistorted,
        mask,
        card_corners,
        width_mm,
        height_mm,
        confidence,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    overlay_path = output_dir / f"{stem}_measurement.png"
    mask_path = output_dir / f"{stem}_mask.png"
    json_path = output_dir / f"{stem}_measurement.json"

    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError("Could not save measurement overlay.")
    if not cv2.imwrite(str(mask_path), mask):
        raise RuntimeError("Could not save casing mask.")

    result = {
        "input_image": str(image_path),
        "width_mm": round(width_mm, 2),
        "height_mm": round(height_mm, 2),
        "segmentation_confidence": round(confidence, 4),
        "probability_threshold": PROBABILITY_THRESHOLD,
        "card_detected": True,
        "card_reference_width_mm": CARD_WIDTH_MM,
        "card_reference_height_mm": CARD_HEIGHT_MM,
        "card_observed_aspect_ratio": round(card_ratio, 4),
        "calibration_file": str(Path(args.calibration)),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "measurement_method": (
            "Undistortion, U-Net segmentation, card-based perspective "
            "rectification, and minimum-area casing rectangle."
        ),
        "caveats": [
            "Assumes casing and reference card lie in the same plane.",
            "Card quadrilateral is estimated from its red pattern region.",
            "Calibration reprojection error is relatively high; "
            "measurements require empirical validation.",
            "Confidence is mean predicted foreground probability, "
            "not a calibrated probability of measurement correctness."
        ],
        "overlay": str(overlay_path),
        "mask": str(mask_path),
    }

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nMEASUREMENT RESULT")
    print(f"Width: {width_mm:.2f} mm")
    print(f"Height: {height_mm:.2f} mm")
    print(f"Segmentation confidence: {confidence:.4f}")
    print(f"Overlay: {overlay_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
