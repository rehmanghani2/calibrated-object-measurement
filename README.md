# Measure It — Segmentation and Metric Measurement

A computer-vision pipeline for segmenting a mobile phone casing and estimating its physical width and height in millimetres using camera calibration and a known-size reference card.

## 1. Project overview

This project implements an image-to-millimetres workflow:

1. Calibrate the camera using checkerboard images.
2. Undistort input images using saved camera parameters.
3. Segment the phone casing using a custom-trained U-Net.
4. Detect a known-size playing card in the image.
5. Rectify the reference plane using a perspective transformation.
6. Estimate casing width and height in millimetres.
7. Save measurement outputs, masks, overlays and evaluation results.

**Object:** Black mobile phone casing.

**Reference object:** Playing card, measured dimensions 56.5 mm × 87.0 mm.

**Casing ground truth:** 78.0 mm × 160.2 mm, measured using a ruler.

## 2. Repository structure

```text
calibrated-object-measurement/
│
├── calibration/
│   ├── calibrate_camera.py
│   ├── camera_calibration.npz
│   ├── calibration_report.json
│   └── reprojection_errors.csv
│
├── dataset/
│   ├── originals/
│   ├── undistorted/
│   ├── labels/
│   ├── splits/
│   ├── prepare_images.py
│   ├── auto_label.py
│   └── verify_training_data.py
│
├── models/
│   ├── unet.py
│   └── outputs/
│       ├── best_unet.pt
│       ├── history.csv
│       └── training_config.json
│
├── training/
│   └── train.py
│
├── evaluation/
│   └── evaluate_test.py
│
├── inference/
│   └── measure.py
│
├── measurement/
│   ├── run_accuracy_evaluation.py
│   ├── accuracy/
│   └── results/
│
├── docs/
│   ├── Plan.md
│   ├── Objectives.md
│   └── Definition_of_Done.md
│
├── requirements.txt
└── README.md
```

## 3. Hardware and software

| Component | Specification |
|---|---|
| Camera | Samsung Galaxy J5 (2016) |
| Normalized image resolution | 4128 × 3096 pixels |
| CPU | Intel Core i5-8365U |
| CPU cores | 4 cores / 8 threads |
| RAM | 16 GB |
| GPU | Intel UHD 620 integrated graphics |
| Inference device | CPU |
| Python | 3.10.5 |
| OpenCV | 4.9.0 |
| NumPy | 1.26.4 |
| PyTorch | 2.5.1+cpu |
| Torchvision | 0.20.1+cpu |

CUDA was unavailable. Model training and inference were performed on CPU.

## 4. Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Install the CPU versions of PyTorch and Torchvision if they are not already installed:

```powershell
pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

## 5. Running the pipeline

Run commands from the repository root.

### 5.1 Camera calibration

```powershell
python calibration/calibrate_camera.py
```

The calibration process detects checkerboard corners and estimates the camera matrix and distortion coefficients.

Outputs:

- `calibration/camera_calibration.npz`
- `calibration/calibration_report.json`
- `calibration/reprojection_errors.csv`

Calibration evidence:

- 27 calibration images supplied.
- 26 images successfully detected.
- OpenCV RMS reprojection error: 4.3883 pixels.

The calibration error is relatively high; measurements should therefore be interpreted with caution.

### 5.2 Image preprocessing

```powershell
python dataset/prepare_images.py
```

This stage normalizes image orientation, loads calibration parameters, undistorts images and writes the processed dataset.

### 5.3 Verify eligible training data

```powershell
python dataset/verify_training_data.py
```

The verification checks image and mask readability, non-empty masks, matching dimensions and split separation.

### 5.4 Train the segmentation model

```powershell
python training/train.py
```

The custom U-Net is trained using the training split, with validation data used for checkpoint selection.

The test split is not used during training.

### 5.5 Evaluate segmentation

```powershell
python evaluation/evaluate_test.py
```

The evaluation reports mask IoU, mAP@0.5, mAP@0.5:0.95, pixel precision and pixel recall.

### 5.6 Measure a phone casing

```powershell
python inference/measure.py --image dataset/originals/img_062.jpg
```

Replace the image path with another supported image.

The inference pipeline:

1. Loads and normalizes the image orientation.
2. Undistorts the image.
3. Runs U-Net segmentation.
4. Detects the reference card.
5. Estimates the reference geometry.
6. Rectifies the reference plane.
7. Estimates casing dimensions in millimetres.
8. Saves measurement results and visual outputs.

The system reports width, height and segmentation confidence. It can reject an image when reference geometry is implausible or required detections are missing.

### 5.7 Run measurement accuracy evaluation

```powershell
python measurement/run_accuracy_evaluation.py
```

This evaluates the eligible test images against the ruler-measured casing dimensions and writes per-instance results and aggregate error statistics.

## 6. Dataset summary

The project contains 129 original casing photographs.

The original committed split contains:

| Split | Original images |
|---|---:|
| Train | 90 |
| Validation | 26 |
| Test | 13 |
| **Total** | **129** |

After annotation quality checks, 93 image-mask pairs were eligible for model training and evaluation:

| Split | Eligible images |
|---|---:|
| Train | 64 |
| Validation | 19 |
| Test | 10 |
| **Total** | **93** |

Thirty-six images were excluded because of failed or questionable annotations.

Annotations were generated algorithmically and quality-checked. They should not be described as fully hand-drawn masks.

The original split lists were preserved. The eligible split lists are stored separately in `dataset/splits/`.

## 7. Model architecture and evaluation

A custom U-Net was implemented in PyTorch.

The architecture was selected as a segmentation model that could be trained on the available dataset and executed on CPU hardware. It does not use an Ultralytics YOLO or Roboflow segmentation model.

| Configuration | Value |
|---|---|
| Architecture | Custom U-Net |
| Trainable parameters | 1,942,577 |
| Input resolution | 384 × 288 |
| Input channels | 3 |
| Output channels | 1 |
| Random seed | 42 |
| Best checkpoint epoch | 12 |
| Best validation IoU | 0.9629 |

### Final test metrics

| Metric | Result |
|---|---:|
| Mean mask IoU | 0.9582 |
| mAP@0.5 | 1.0000 |
| mAP@0.5:0.95 | 0.9523 |
| Pixel precision | 0.9687 |
| Pixel recall | 0.9914 |

The mAP metrics use the project's custom single-object segmentation evaluation protocol, not the COCO evaluation protocol.

## 8. Measurement accuracy

Ground truth:

- Width: 78.0 mm
- Height: 160.2 mm

Nine of ten eligible test images produced successful measurements. One image, `img_068.jpg`, was rejected because the detected reference-card aspect ratio was implausible.

### Aggregate results

| Metric | Width | Height |
|---|---:|---:|
| Mean absolute error | 15.3422 mm | 10.1722 mm |
| Median absolute error | 19.56 mm | 12.57 mm |
| Worst absolute error | 26.73 mm | 18.40 mm |
| Mean absolute percentage error | 19.6695% | 6.3497% |

These are repeated photographs of one physical casing, not ten independent physical objects.

The width measurement error is substantial. The segmentation scores alone do not establish metric measurement accuracy.

## 9. Model weights and reproducibility

Model checkpoint:

`models/outputs/best_unet.pt`

SHA-256:

```text
3765113A1266C7DA884285352D524FE5578EA2AC11CF3307D34C9D410323DB29
```

The checkpoint was selected using validation performance. The held-out test split was reserved for final evaluation.

Pinned core dependencies:

```text
numpy==1.26.4
opencv-python==4.9.0.80
```

PyTorch and Torchvision versions are documented above.

## 10. Known limitations

1. Calibration reprojection error is relatively high at 4.3883 pixels.
2. The card quadrilateral is estimated from its red patterned region rather than exact outer card edges.
3. Measurement assumes the casing and reference card lie in the same plane.
4. Perspective, tilt, card detection and reference geometry can affect scale estimation.
5. Segmentation confidence is the mean predicted foreground probability, not a calibrated probability of measurement correctness.
6. Width measurement errors remain high.
7. The accuracy evaluation contains repeated photographs of one physical casing, not multiple independent objects.
8. Raw-versus-undistorted measurement comparison has not been completed.
9. A separately defined speed or accuracy target has not been evaluated.
10. A controlled breaking-condition experiment has not been completed.
11. The refusal threshold and refusal cost have not been comprehensively evaluated.

## 11. Assessment status

The calibration, dataset preparation, model training, segmentation evaluation and core measurement pipeline have been implemented.

Measurement accuracy evaluation is partial: nine successful measurements were recorded and one reference-geometry rejection was retained.

The raw-versus-undistorted comparison, separate performance target and controlled breaking-condition experiment remain incomplete.

See:

- `docs/Plan.md`
- `docs/Objectives.md`
- `docs/Definition_of_Done.md`

These documents provide the project decisions, measured results and outstanding requirements.