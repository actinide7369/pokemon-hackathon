Pokemon PS — CV–NLP Pipeline
=================================

Project overview
----------------

Pokemon PS is an end-to-end computer-vision + natural-language pipeline designed to perform robust 4-class detection and precise target extraction from noisy, adversarial prompts. The system was developed as part of a simulated competition setting where scoring penalties encourage conservative, confidence-aware outputs. The pipeline pairs a high-capacity detector with a fine-tuned transformer extractor and a score-aware decision heuristic.

Core components
---------------

- **Detector:** YOLOv8x — one-stage object detector used for class bounding-box proposals and high-throughput inference.
- **Extractor:** DeBERTa-v3-small — transformer encoder fine-tuned with an attention-pooling head to extract target tokens from noisy prompts and multimodal context.
- **Filter & clustering stack:** class-specific CV prefilters (color masks, edge-density thresholds, morphological operations) followed by `DBSCAN` clustering to consolidate fragmented detections.
- **Heuristic controller:** score-aware decision logic that uses confidence thresholds and a confidence-gated buffer of alternative "shots" to minimize competition penalties.

Key techniques and engineering
------------------------------

- **Class-specific CV filtering:** For each class we implemented lightweight prefilters that exploit dominant color properties and edge-density heuristics to remove obvious false positives before invoking the detector. This reduces runtime and improves precision upstream.
- **Edge-density and morphological heuristics:** Local edge-density maps and morphological cleanup help stabilize detections in cluttered backgrounds and under varying lighting.
- **Spatial consolidation via DBSCAN:** Instead of greedy NMS alone, we cluster edge/keypoint responses and detection proposals with `DBSCAN` to merge fragmented or pose-separated detections robustly.
- **Pose-variant augmentation:** To handle extreme viewpoints and foreshortening we synthesized pose-warped images with randomized rotations, perspective transforms, and elastic deformations during detector training.
- **Adversarial prompt generation:** We generated 10,000 synthetic adversarial prompts (typos, paraphrases, inserted noise tokens, format shifts) to simulate noisy human inputs and prompt-engineering attacks.
- **Transformer fine-tuning with attention pooling:** `DeBERTa-v3-small` was fine-tuned on a mixed corpus of real and synthetic prompts. An attention-pooling head aggregates token-level signals into a single target representation for extraction and classification.
- **Score-aware, confidence-gated buffer shots:** A buffer retains top-k alternate predictions. The emission policy consults a scoring heuristic tuned to the contest penalty rules: emit conservative, high-confidence outputs when penalty is high; use buffered alternatives when the expected net gain favors recall.

Data and training
-----------------

- **Training data:** mix of annotated images for the four target classes and a synthetic prompt dataset of 10K adversarial/noisy prompts aligned with image candidates.
- **Staged training regime:**
  - Pretrain and augment the detector on heavy visual augmentations (pose, color jitter, adversarial occlusions).
  - Generate image-prompt pairs (real + synthetic) and fine-tune the `DeBERTa-v3-small` extractor on noisy prompt extraction tasks.
  - Joint validation where detector proposals are fed into the extractor to measure end-to-end performance and to tune the scoring heuristic.
- **Losses & objectives:** detector uses standard object-detection losses (box regression + classification), extractor trained with cross-entropy / sequence labeling loss adapted for target extraction, and a scoring calibration stage that optimizes expected leaderboard score under simulated penalties.

Inference pipeline
------------------

1. Input image is run through class-specific CV prefilters (color masks, edge-density checks).
2. Filtered image passes to `YOLOv8x` detector for bounding-box proposals.
3. Proposal candidates are consolidated via `DBSCAN` clustering and lightweight ranking.
4. Top candidates plus the noisy prompt are forwarded to `DeBERTa-v3-small` extractor to determine the final target token(s).
5. Scoring heuristic evaluates confidence and consults the buffer of alternative shots; it decides whether to emit the primary prediction or a buffered alternative to optimize expected net score.

Evaluation
----------

- **Detection metrics:** mAP, class-wise precision and recall.
- **Extraction metrics:** F1, exact-match, and character-level recall where applicable.
- **Leaderboard-aware expected-score simulation:** simulates the contest penalty rules (false positive/false negative penalties) to evaluate the net effect of heuristics and buffer policies.
- **Ablation studies:** measured the impact of class-specific filters, DBSCAN clustering, buffer size, and confidence thresholds on final expected score.

Best practices and reproducibility
---------------------------------

- All augmentations and the synthetic prompt generator are seeded to allow deterministic reproduction of experiments.
- Recommended environment: Python 3.8+, CUDA-enabled GPU for detector training, and a compact transformer inference environment for low-latency deployments.
- Core scripts: `train_cv.py` (detector training) and `train_nlp.py` (extractor training). See Quick Start for example commands.

Quick start
-----------

Install dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

Train detector (example):

```bash
python train_cv.py --config configs/yolov8x_config.yaml --data data/img_dataset
```

Train extractor (example):

```bash
python train_nlp.py --data data/prompt_pairs --model deberta-v3-small
```

Inference example (pseudo-command):

```bash
python inference.py --image input.jpg --prompt "noisy prompt text"
```

Design rationale & trade-offs
---------------------------

- Choosing `YOLOv8x` provides a strong accuracy baseline while keeping inference latency manageable; pairing a compact transformer (`DeBERTa-v3-small`) balances extraction quality and deployment cost.
- Class-specific prefilters trade a small amount of engineering complexity for notable gains in precision and runtime efficiency by avoiding full detection on obvious negatives.
- The buffer/heuristic system explicitly optimizes for contest scoring — in standard applications you can simplify or remove the buffer in favor of strict confidence thresholds.

Resume-friendly summary
-----------------------

Built an end-to-end CV–NLP pipeline combining `YOLOv8x` and `DeBERTa-v3-small` for 4-class detection and target extraction from noisy prompts. Engineered class-specific CV filters (color masking, edge-density), `DBSCAN` spatial consolidation, and pose-variant augmentation. Generated 10K synthetic adversarial prompts and fine-tuned a transformer with an attention-pooling head for robust target extraction. Designed a score-aware, confidence-gated buffer-shot heuristic to exploit competition penalty scoring and improve net leaderboard performance.

Contact & licensing
-------------------

This repository contains illustrative research code. Use or redistribution should conform to upstream model licenses (YOLO/DeBERTa) and any included datasets. 
