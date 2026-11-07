# Plan — Measure It: Segmentation and Metric Measurement

## Project objective

Build a computer-vision pipeline that uses a calibrated camera, a trained segmentation model, and a known-size reference object to estimate the width and height of a phone casing in millimetres.

The project uses a Samsung Galaxy J5 (2016) camera and a black phone casing. A playing card is used as the in-image dimensional reference.

## Execution plan and actual progress

| Stage | Planned work | Actual result |
|---|---|---|
| 1. Calibration | Capture checkerboard images, calibrate with OpenCV, save camera parameters and report reprojection error. | Completed. 26 of 27 calibration images detected. RMS reprojection error: 4.3883 px. |
| 2. Dataset | Capture, undistort, label and split at least 70 images. | 129 original images processed. 93 eligible image-mask pairs retained: 64 train, 19 validation, 10 test. |
| 3. Model | Train a permitted segmentation architecture and evaluate on the held-out test split. | Custom U-Net trained. Best validation IoU: 0.9629. Final test metrics recorded. |
| 4. Measurement pipeline | Implement one-command image-to-millimetres inference, including undistortion, segmentation, reference detection and output generation. | Implemented and tested. Missing or implausible reference geometry can cause measurement rejection. |
| 5. Measurement accuracy | Compare system measurements against ruler measurements for at least 10 instances. | Partially completed. 9 of 10 original test images produced measurements. One was rejected. |
| 6. Undistortion comparison | Compare raw-image and undistorted-image measurement errors on the same instances. | Not completed. |
| 7. Performance objective | Measure one selected speed or accuracy target. | Not completed as a separately defined and measured objective. |
| 8. Breaking condition | Quantify error under one selected adverse condition. | Not completed. |
| 9. Refusal behaviour | Demonstrate when the system should abstain from reporting a measurement and quantify refusal cost. | A reference-geometry rejection was observed, but a complete threshold and refusal-cost evaluation was not completed. |
| 10. Decision record | Record the chosen approach, rejected alternative and cost. | Key implementation decisions and limitations are documented below; a complete formal decision-record evaluation was not completed. |

## Implementation decisions

- **Calibration:** OpenCV checkerboard calibration was used. The measured checkerboard square size was 25 mm.
- **Segmentation:** A custom U-Net was selected instead of an Ultralytics YOLO or Roboflow model.
- **Annotation:** Initial masks were generated algorithmically and then quality-checked. Images with failed or questionable masks were excluded rather than treated as reliable ground truth.
- **Dataset split:** The original train/validation/test file lists were committed before model training. The eligible subsets were generated without changing those original lists.
- **Measurement reference:** The playing card was measured as 56.5 mm × 87.0 mm. The casing ruler measurement was 78.0 mm × 160.2 mm.
- **Inference:** The system undistorts the image, predicts the casing mask, detects the card reference and uses a perspective transformation to estimate dimensions.

## Limitations and decisions

The calibration reprojection error is relatively high. The current calibration was retained because there was insufficient time to recapture and recalibrate.

The annotation workflow used algorithmic mask generation followed by quality checks rather than fully manual tracing of every mask. This saved annotation time but leaves a risk of label noise.

The card detector estimates the reference quadrilateral from its red patterned region. It does not directly identify the exact outer card boundary. This can introduce dimensional scale error.

The measurement results show substantial width errors despite high segmentation confidence. Segmentation confidence is the mean predicted foreground probability; it is not a calibrated probability that the final physical measurement is correct.

## Outstanding work

- Complete a 10-instance measurement accuracy evaluation.
- Compare raw and undistorted measurements.
- Define and evaluate a speed or accuracy target.
- Quantify one breaking condition.
- Complete a refusal-threshold and refusal-cost evaluation.
- Complete the final recording and submission steps.

The unfinished work is disclosed rather than represented as completed.