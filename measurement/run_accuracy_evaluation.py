
from pathlib import Path
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_WIDTH_MM = 78.0
GROUND_TRUTH_HEIGHT_MM = 160.2

TEST_LIST = ROOT / "dataset" / "splits" / "test_eligible.txt"
IMAGE_DIR = ROOT / "dataset" / "originals"
INFERENCE_SCRIPT = ROOT / "inference" / "measure.py"

OUT_DIR = ROOT / "measurement" / "accuracy"
LOG_DIR = OUT_DIR / "run_logs"
RESULTS_DIR = ROOT / "measurement" / "results"


def calculate_metrics(rows, dimension):
    error_key = f"{dimension}_absolute_error_mm"
    percentage_key = f"{dimension}_absolute_percentage_error"

    errors = [row[error_key] for row in rows]
    percentages = [row[percentage_key] for row in rows]

    worst_index = max(range(len(rows)), key=lambda i: errors[i])

    return {
        "mae_mm": round(statistics.mean(errors), 4),
        "median_absolute_error_mm": round(statistics.median(errors), 4),
        "worst_absolute_error_mm": round(errors[worst_index], 4),
        "worst_case_image": rows[worst_index]["image"],
        "mean_absolute_percentage_error": round(
            statistics.mean(percentages), 4
        ),
        "max_absolute_percentage_error": round(max(percentages), 4),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_LIST.exists():
        raise FileNotFoundError(f"Test list not found: {TEST_LIST}")

    if not INFERENCE_SCRIPT.exists():
        raise FileNotFoundError(f"Inference script not found: {INFERENCE_SCRIPT}")

    image_names = [
        line.strip()
        for line in TEST_LIST.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if len(image_names) < 10:
        raise RuntimeError(
            f"Expected at least 10 test images; found {len(image_names)}"
        )

    if len(image_names) != len(set(image_names)):
        raise RuntimeError("Duplicate entries found in test split.")

    # Preserve the exact list used for this evaluation.
    (OUT_DIR / "test_images_used.txt").write_text(
        "\n".join(image_names) + "\n",
        encoding="utf-8"
    )

    rows = []
    failures = []

    print(f"Running final accuracy evaluation on {len(image_names)} images.")
    print(f"Ground truth: {GROUND_TRUTH_WIDTH_MM} x "
          f"{GROUND_TRUTH_HEIGHT_MM} mm")
    print("The test images are not used for training or tuning.\n")

    for index, image_name in enumerate(image_names, start=1):
        image_path = IMAGE_DIR / image_name
        log_path = LOG_DIR / f"{Path(image_name).stem}_console.txt"
        json_path = RESULTS_DIR / f"{Path(image_name).stem}_measurement.json"

        print(f"[{index}/{len(image_names)}] {image_name}")

        if not image_path.exists():
            failures.append({
                "image": image_name,
                "reason": f"Image not found: {image_path}"
            })
            print("  FAILED: input image not found")
            continue

        # Run the existing measurement pipeline without changing it.
        command = [
            sys.executable,
            str(INFERENCE_SCRIPT),
            "--image",
            str(image_path)
        ]

        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        log_text = (
            "COMMAND:\n" + subprocess.list2cmdline(command)
            + "\n\nRETURN CODE:\n" + str(completed.returncode)
            + "\n\nSTDOUT:\n" + completed.stdout
            + "\n\nSTDERR:\n" + completed.stderr
        )
        log_path.write_text(log_text, encoding="utf-8")

        if completed.returncode != 0:
            failures.append({
                "image": image_name,
                "reason": f"Inference returned exit code {completed.returncode}"
            })
            print("  FAILED: inference command")
            continue

        if not json_path.exists():
            failures.append({
                "image": image_name,
                "reason": f"Expected measurement JSON not found: {json_path}"
            })
            print("  FAILED: output JSON not found")
            continue

        try:
            result = json.loads(json_path.read_text(encoding="utf-8"))

            if not result.get("card_detected", False):
                raise ValueError("Reference card was not detected.")

            if result.get("input_image") is None:
                raise ValueError("Output JSON has no input_image field.")

            width = float(result["width_mm"])
            height = float(result["height_mm"])
            confidence = float(result["segmentation_confidence"])

            if width <= 0 or height <= 0:
                raise ValueError("Non-positive measurement.")

            width_error = abs(width - GROUND_TRUTH_WIDTH_MM)
            height_error = abs(height - GROUND_TRUTH_HEIGHT_MM)

            row = {
                "image": image_name,
                "ground_truth_width_mm": GROUND_TRUTH_WIDTH_MM,
                "measured_width_mm": round(width, 4),
                "width_absolute_error_mm": round(width_error, 4),
                "width_absolute_percentage_error": round(
                    width_error / GROUND_TRUTH_WIDTH_MM * 100, 4
                ),
                "ground_truth_height_mm": GROUND_TRUTH_HEIGHT_MM,
                "measured_height_mm": round(height, 4),
                "height_absolute_error_mm": round(height_error, 4),
                "height_absolute_percentage_error": round(
                    height_error / GROUND_TRUTH_HEIGHT_MM * 100, 4
                ),
                "segmentation_confidence": round(confidence, 4),
                "card_detected": result["card_detected"],
                "measurement_json": str(
                    json_path.relative_to(ROOT)
                ).replace("\\", "/"),
                "console_log": str(
                    log_path.relative_to(ROOT)
                ).replace("\\", "/"),
            }

            rows.append(row)

            print(
                f"  Width {width:.2f} mm "
                f"(error {width_error:.2f} mm); "
                f"Height {height:.2f} mm "
                f"(error {height_error:.2f} mm); "
                f"Confidence {confidence:.4f}"
            )

        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            failures.append({
                "image": image_name,
                "reason": str(exc)
            })
            print(f"  FAILED: invalid measurement output: {exc}")

    # Save all successfully evaluated instances.
    csv_path = OUT_DIR / "per_instance_results.csv"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "Final held-out test measurement accuracy",
        "test_images_requested": len(image_names),
        "successful_measurements": len(rows),
        "failed_measurements": len(failures),
        "ground_truth": {
            "width_mm": GROUND_TRUTH_WIDTH_MM,
            "height_mm": GROUND_TRUTH_HEIGHT_MM,
            "source": "User's ruler measurement of the physical casing"
        },
        "dataset_note": (
            "These are repeated photographic instances of one physical casing, "
            "not independent physical objects."
        ),
        "width": calculate_metrics(rows, "width") if rows else None,
        "height": calculate_metrics(rows, "height") if rows else None,
        "failures": failures,
        "per_instance_csv": "measurement/accuracy/per_instance_results.csv",
    }

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    # Human-readable report for the repository.
    report = [
        "# Measurement Accuracy Evaluation",
        "",
        f"- Successful measurements: {len(rows)} / {len(image_names)}",
        f"- Failed measurements: {len(failures)}",
        "- Evaluation set: committed eligible test split",
        "- Ground truth: 78.0 mm width × 160.2 mm height",
        "- Ground truth method: ruler measurement of the physical casing",
        "- Object count: one physical casing, photographed repeatedly",
        "",
        "## Per-instance results",
        "",
        "| Image | GT width | Measured width | Width error | GT height | Measured height | Height error | Confidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        report.append(
            f"| {row['image']} "
            f"| {row['ground_truth_width_mm']:.2f} "
            f"| {row['measured_width_mm']:.2f} "
            f"| {row['width_absolute_error_mm']:.2f} "
            f"| {row['ground_truth_height_mm']:.2f} "
            f"| {row['measured_height_mm']:.2f} "
            f"| {row['height_absolute_error_mm']:.2f} "
            f"| {row['segmentation_confidence']:.4f} |"
        )

    if rows:
        width = summary["width"]
        height = summary["height"]

        report.extend([
            "",
            "## Aggregate errors",
            "",
            "| Metric | Width | Height |",
            "|---|---:|---:|",
            f"| MAE (mm) | {width['mae_mm']:.4f} | {height['mae_mm']:.4f} |",
            f"| Median absolute error (mm) | {width['median_absolute_error_mm']:.4f} | {height['median_absolute_error_mm']:.4f} |",
            f"| Worst absolute error (mm) | {width['worst_absolute_error_mm']:.4f} | {height['worst_absolute_error_mm']:.4f} |",
            f"| Worst-case image | {width['worst_case_image']} | {height['worst_case_image']} |",
            f"| Mean absolute percentage error (%) | {width['mean_absolute_percentage_error']:.4f} | {height['mean_absolute_percentage_error']:.4f} |",
            "",
            "## Evaluation limitations",
            "",
            "- The ten images show the same physical casing; they are not ten independent objects.",
            "- The same ruler-measured dimensions are used as ground truth for each photograph.",
            "- Results depend on the accuracy of the ruler measurements and the assumption that the casing and reference card lie in the same plane.",
            "- Segmentation confidence is not a calibrated probability of measurement correctness.",
            "- Failed measurements are listed in summary.json and are not silently counted as successful measurements.",
        ])

    if failures:
        report.extend(["", "## Failed instances", ""])
        for failure in failures:
            report.append(
                f"- {failure['image']}: {failure['reason']}"
            )

    report_path = OUT_DIR / "accuracy_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n========== ACCURACY EVALUATION SUMMARY ==========")
    print(f"Successful: {len(rows)} / {len(image_names)}")
    print(f"Failed: {len(failures)}")

    if rows:
        print(
            f"Width MAE: {summary['width']['mae_mm']:.4f} mm | "
            f"Median: {summary['width']['median_absolute_error_mm']:.4f} mm | "
            f"Worst: {summary['width']['worst_absolute_error_mm']:.4f} mm | "
            f"Mean absolute percentage error: "
            f"{summary['width']['mean_absolute_percentage_error']:.4f}%"
        )
        print(
            f"Height MAE: {summary['height']['mae_mm']:.4f} mm | "
            f"Median: {summary['height']['median_absolute_error_mm']:.4f} mm | "
            f"Worst: {summary['height']['worst_absolute_error_mm']:.4f} mm | "
            f"Mean absolute percentage error: "
            f"{summary['height']['mean_absolute_percentage_error']:.4f}%"
        )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    print(f"  {report_path}")
    print(f"  {LOG_DIR}")

    if failures:
        print("\nSome instances failed. Review summary.json before reporting results.")
        raise SystemExit(1)

    if len(rows) != len(image_names):
        raise SystemExit("Evaluation incomplete: not all test images succeeded.")

    print("\nPASS: all listed test images were evaluated.")


if __name__ == "__main__":
    main()
