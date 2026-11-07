# Measure It — Project Plan

**Repository:** `calibrated-object-measurement`
**Branch:** `dev-test01`
**Time limit:** 8 hours

## Goal
Build a system that segments the mobile-phone casing and measures its outer width and height in millimetres using the known-size playing card in each photograph.

## Work plan

| Time | Planned work |
|---|---|
| 0:00–0:30 | Verify camera settings, reference dimensions and environment. |
| 0:30–1:15 | Calibrate camera using 20+ checkerboard images and OpenCV. |
| 1:15–2:45 | Undistort images, label at least 70 casing photos and confirm fixed splits. |
| 2:45–4:30 | Train and evaluate the segmentation model. |
| 4:30–5:45 | Implement one-command image-to-millimetres measurement and failure handling. |
| 5:45–6:45 | Compare system and calliper measurements on 10+ instances; compare raw and undistorted errors. |
| 6:45–7:30 | Test one breaking condition and measurement abstention. |
| 7:30–8:00 | Finalise documentation, reproducibility checks and Loom recording. |

## Approach

Use OpenCV for calibration, undistortion and card detection. Train a binary U-Net with a ResNet-34 encoder using PyTorch. Segment the casing as one class; use the playing card only as the scale reference. Use the 129 collected casing photographs and 27 available checkerboard photographs. Verify physical reference dimensions and checkerboard square size with callipers.

## Deliberately skipped

- Additional object classes or multi-object detection.
- A mobile application, web interface or deployment infrastructure.
- Real-time video processing and production optimisation.
- Alternative model architectures unless the selected approach proves unsuitable.

Prioritise assessment items 1–4. Do not start item 8 before item 5 is complete. Keep the test split isolated from training and tuning. If time runs short, defer optional work and document incomplete requirements honestly.