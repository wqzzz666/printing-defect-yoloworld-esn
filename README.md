# printing-defect-yoloworld-esn
# Printing-defect inspection reproducibility package (YOLO-World + CLIP + ESN)

This repository provides the code and minimal resources to reproduce the main pipeline described in our manuscript:
(1) ROI detection and cropping using a fine-tuned YOLO-World detector, and
(2) similarity scoring using the proposed Enhanced Siamese Network (ESN).

## What is released
- Source code (training / inference / evaluation / runtime benchmark scripts).
- A YOLO-World checkpoint (`best.pt`) for ROI detection and cropping.
- A de-identified subset of 185 original images (`subset185_images.zip`) without derived augmentations.

All binary assets are provided in the GitHub Release v1.0:
https://github.com/wqzzz666/printing-defect-yoloworld-esn/releases/tag/v1.0

## Notes on data and checkpoints
- The full industrial dataset and derived augmented images cannot be redistributed.
- The ESN checkpoint from the original experiments was not preserved due to storage limitations at the time.
  ESN can be reproduced by re-training using the released code and configuration.

## Environment
Tested with:
- Python 3.8.10
- NVIDIA GeForce RTX 3070 GPU

Install dependencies:
```bash
pip install -r requirements.txt
