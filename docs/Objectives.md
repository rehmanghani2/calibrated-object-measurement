# Objectives — Measure It

## Measurement setup

- **Camera:** Samsung Galaxy J5 (2016)
- **Image resolution after orientation normalization:** 4128 × 3096 pixels
- **Object:** Black mobile phone casing
- **Reference:** Playing card, measured as 56.5 mm × 87.0 mm
- **Casing ground truth:** 78.0 mm width × 160.2 mm height, measured with a ruler
- **Compute:** Intel Core i5-8365U, 16 GB RAM, Intel UHD 620 integrated graphics; CPU inference
- **Python:** 3.10.5
- **OpenCV:** 4.9.0
- **PyTorch:** 2.5.1+cpu
- **Random seed:** 42

## MO-1 — Measurement accuracy

**What is measured**

Absolute and percentage differences between the system's estimated casing dimensions and the ruler-measured dimensions.

**Method**

The measurement evaluation script runs the inference pipeline on the committed eligible test images. It compares each successful system result against the casing ground truth of 78.0 mm × 160.2 mm.

**Ground truth**

| Dimension | Ground truth |
|---|---:|
| Width | 78.0 mm |
| Height | 160.2 mm |

**Results**

Nine of the ten eligible test images produced a measurement. `img_068.jpg` was rejected because the detected reference-card geometry had an implausible aspect ratio of 1.962.

| Image | Measured width (mm) | Absolute width error (mm) | Measured height (mm) | Absolute height error (mm) |
|---|---:|---:|---:|---:|
| img_062.jpg | 79.27 | 1.27 | 166.61 | 6.41 |
| img_033.jpg | 89.64 | 11.64 | 155.89 | 4.31 |
| img_030.jpg | 104.73 | 26.73 | 157.72 | 2.48 |
| img_094.jpg | 58.44 | 19.56 | 145.85 | 14.35 |
| img_127.jpg | 100.69 | 22.69 | 178.60 | 18.40 |
| img_092.jpg | 52.73 | 25.27 | 146.17 | 14.03 |
| img_035.jpg | 98.62 | 20.62 | 172.77 | 12.57 |
| img_024.jpg | 84.31 | 6.31 | 174.58 | 14.38 |
| img_089.jpg | 74.01 | 3.99 | 155.58 | 4.62 |

**Aggregate results — successful measurements only**

| Metric | Width | Height |
|---|---:|---:|
| Mean absolute error (MAE) | 15.3422 mm | 10.1722 mm |
| Median absolute error | 19.56 mm | 12.57 mm |
| Worst absolute error | 26.73 mm | 18.40 mm |
| Mean absolute percentage error (MAPE) | 19.6695% | 6.3497% |

**Interpretation**

The measurement pipeline executes, but its metric accuracy is not yet satisfactory. Width estimates show particularly large errors. High segmentation confidence does not guarantee correct metric scale or reference geometry.

These measurements are repeated photographs of one physical casing, not measurements of ten independent physical objects. The evaluation therefore does not establish generalization across different casings.

The required target value for MO-1 was not preserved in the established results. Consequently, target attainment is not claimed.

## MO-2 — Segmentation quality

**What is measured**

Mask overlap and pixel-level classification performance on the held-out eligible test split.

**Method**

The trained U-Net checkpoint was evaluated on the ten images in the eligible test split. The test split was not used for training or checkpoint selection.

**Results**

| Metric | Test result |
|---|---:|
| Mean mask IoU | 0.9582 |
| mAP@0.5 | 1.0000 |
| mAP@0.5:0.95 | 0.9523 |
| Pixel precision | 0.9687 |
| Pixel recall | 0.9914 |

The mAP values use the project's custom single-object segmentation evaluation protocol. They should not be interpreted as COCO benchmark results.

**Training information**

- Architecture: Custom U-Net
- Trainable parameters: 1,942,577
- Input resolution: 384 × 288
- Training images: 64
- Validation images: 19
- Test images: 10
- Best validation IoU: 0.9629 at epoch 12
- Early stopping: training stopped after epoch 17

## Calibration objective evidence

- Calibration images detected: 26 of 27
- OpenCV RMS reprojection error: 4.3883195 px
- Mean per-image reprojection RMS: 4.219738 px
- Calibration parameters saved in `calibration/camera_calibration.npz`

The reprojection error is relatively high and is an important limitation of the measurement system.

## Objectives not evaluated

The following objectives remain incomplete:

1. Raw-image versus undistorted-image measurement-error comparison.
2. A separately defined and measured speed or accuracy target.
3. Error measurement under a selected breaking condition.
4. A complete refusal-threshold and refusal-cost evaluation.

No results are claimed for these objectives.