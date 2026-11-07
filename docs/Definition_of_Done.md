# Definition of Done --- Measure It Assessment

**Project:** `calibrated-object-measurement`\
**Status:** Checklist to be completed with evidence. Do not tick an item
until its evidence exists.

## A. Planning and reproducibility

-   [ ] `docs/Plan.md`, `docs/Objectives.md` and
    `docs/Definition_of_Done.md` reviewed and committed before
    implementation.
-   [ ] Root `README.md` documents setup, commands, hardware, camera,
    dataset summary and reproduction steps.
-   [ ] Work is on branch `dev-test01`.
-   [ ] Git history contains progressive, meaningful commits; no single
    end-of-project commit or premature finished-pipeline first commit.
-   [ ] Python and system dependencies are pinned or version-recorded;
    random seeds and training configuration are recorded.
-   [ ] Train/validation/test lists are committed before training and
    remain unchanged during test evaluation.
-   [ ] Dataset origin and ownership are documented; only self-captured
    project images are used.
-   [ ] Large dataset/model artifacts have stable documented locations
    and SHA-256 checksums where they are not stored in Git.

## B. Camera calibration

-   [ ] At least 20 checkerboard calibration images are available;
    project inventory currently reports 27.
-   [ ] Checkerboard pattern is configured correctly: 7 × 9 inner
    corners for an 8 × 10-square board.
-   [ ] Printed square size is physically verified with callipers; the
    reported 25 mm value is not assumed without verification.
-   [ ] Camera capture settings and calibration image resolution are
    recorded.
-   [ ] OpenCV calibration code detects corners and estimates camera
    matrix and distortion coefficients.
-   [ ] Camera matrix, distortion coefficients and calibration image
    size are saved in a reusable file.
-   [ ] Reprojection error is calculated and reported from actual
    calibration output.
-   [ ] Undistortion is implemented and checked on representative
    images.

## C. Dataset and annotation

-   [ ] All 129 original casing JPGs are preserved in
    `dataset/originals/`; source backup remains unchanged.
-   [ ] Original images are undistorted into the documented
    processed-image location.
-   [ ] At least 70 project photographs have corresponding
    human-reviewed binary segmentation masks; target is to label all 129
    if time permits.
-   [ ] Annotation convention is documented: one casing class; include
    the raised camera housing as part of the outer casing silhouette.
-   [ ] Masks are binary, aligned to their corresponding images, and
    checked for dimensions, empty masks and filename mismatches.
-   [ ] Automatic annotation proposals, if used, are reviewed and
    corrected by a human.
-   [ ] Dataset manifest and fixed split lists are present.
-   [ ] Split counts, uniqueness and overlap are checked;
    near-duplicate/correlated captures across splits are reviewed and
    documented.
-   [ ] Test images and masks are not used for training, model
    selection, threshold tuning or iterative development.

## D. Segmentation model

-   [ ] The final model is trained/fine-tuned on the project's own
    labelled images.
-   [ ] The implementation does not use Ultralytics YOLO or a Roboflow
    model.
-   [ ] Model architecture, encoder, loss, optimizer, seed,
    preprocessing, hyperparameters and software versions are recorded.
-   [ ] Training and validation histories are saved.
-   [ ] Best checkpoint is selected using validation data only.
-   [ ] Test mask IoU, precision, recall, mAP@0.5 and mAP@0.5:0.95 are
    reported.
-   [ ] Binary segmentation mAP conversion and matching protocol are
    explicitly defined.
-   [ ] Model weights are accessible to a reviewer, with a SHA-256
    checksum if hosted externally.

## E. One-command image-to-millimetre pipeline

-   [ ] One documented command accepts an input image and runs the full
    pipeline.
-   [ ] Pipeline loads calibration parameters and undistorts the image.
-   [ ] Pipeline segments the casing and detects the playing-card
    reference.
-   [ ] Card dimensions are physically verified and configured as width
    56.5 mm × height 87.0 mm only after verification.
-   [ ] Card rectangle fitting uses intact edges and handles the
    clipped/notched corner.
-   [ ] Casing width and height are returned in millimetres with
    confidence/quality information.
-   [ ] An output overlay shows the casing mask/contour, reference
    detection and measurement annotations.
-   [ ] Missing casing, missing/invalid reference, poor segmentation and
    unsuitable geometry produce explicit errors or abstentions rather
    than fabricated measurements.
-   [ ] The README includes exact setup and run commands.

## F. Manual measurement evaluation

-   [ ] At least 10 measurement instances are evaluated by both the
    system and manual calliper measurement.
-   [ ] Manual ground truth is recorded with measurement tool, date and
    units.
-   [ ] A per-instance table reports ground truth, prediction, absolute
    error and percentage error for width and height.
-   [ ] MAE, mean percentage error/MAPE, median and worst error are
    calculated from the recorded instances.
-   [ ] The meaning of an "instance" is described accurately; repeated
    images of one physical casing are not misrepresented as different
    physical objects.
-   [ ] Raw-image and undistorted-image measurements are compared on the
    same instances, with error changes quantified.

## G. Performance, robustness and abstention

-   [ ] At least one speed or accuracy target from `Objectives.md` is
    measured and reported.
-   [ ] Five timing runs are recorded for the end-to-end pipeline;
    median and spread are reported with hardware/software context.
-   [ ] One breaking condition (for example tilt, distance, lighting or
    proximity to image edge) is selected and quantified.
-   [ ] Breaking-condition results are collected only after the
    manual-versus-system evaluation is complete.
-   [ ] Abstention criteria are defined using validation data, not the
    test set.
-   [ ] Refused and accepted sample counts, refusal rate/coverage and
    accepted-case error are reported.
-   [ ] The cost of refusal and the risk of an incorrect accepted
    measurement are discussed.

## H. Final evidence and handoff

-   [ ] Results are reproducible from documented commands and saved
    configuration.
-   [ ] README lists OS, CPU, RAM, GPU/compute route, camera model and
    capture resolution.
-   [ ] Final report includes calibration, dataset, model metrics,
    measurement errors, raw-versus-undistorted comparison, performance,
    breaking condition and abstention results.
-   [ ] Loom recording is no longer than five minutes and demonstrates a
    newly captured image, command execution, physical ruler/calliper
    comparison, code trace and answers to assessment questions K1--K4.
-   [ ] Final Git history, repository contents, external artifact links
    and checksums are reviewed.
-   [ ] Every checked item has an identifiable file, command output,
    metric table, commit or recording as evidence.

## Known items not yet evidenced

-   Physical calliper verification of checkerboard square size and
    playing-card dimensions.
-   Actual camera calibration parameters and reprojection error.
-   Human-reviewed segmentation masks and trained model results.
-   Physical calliper ground truth for the measurement trials.
-   Runtime, robustness and abstention measurements.
-   Final README and Loom recording.

These are outstanding tasks, not completed results.
