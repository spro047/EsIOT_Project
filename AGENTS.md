# EsIOT Project — AGENTS.md

## Project type
Research / Jupyter-notebook-driven smart agriculture project. **Not a distributable package.** No build system, no test framework, no CI, no linter/formatter/typechecker.

## Repository structure
| Path | Role |
|------|------|
| `SHASHANK/` | Computer Vision — MobileNetV2 plant disease classifier (39 classes, 14 crops) |
| `SHASHANK/scripts/` | Python scripts (training, conversion, inference) |
| `SHASHANK/models/` | Model checkpoints (`.pt`, `.keras`) — gitignored |
| `SHASHANK/notebooks/` | Jupyter notebooks (training, EDA, testing) |
| `SHASHANK/docs/` | Architecture docs and reports |
| `ANUSHRI/` | Soil health & crop recommendation — Random Forest via scikit-learn |
| `SRUSHTI/` | Research & dataset curation (mostly reference materials) |
| `TANVI/` | IoT integration — **stub** (only `.gitkeep`) |
| `DATASET/` | Image datasets (~55K original, ~61K augmented) — **gitignored** |
| `Papers/` | Reference PDFs, slides, report, and a **`.pem` key that should not be committed** |

## Virtual environments (two)
- `.venv\` — Python 3.14.3, for PyTorch / sklearn / pandas workflows
- `.venv_tf\` — Python 3.12.10, for TensorFlow workflows (TF lacks Python 3.14 support)
No `requirements.txt` in either. No lockfiles. Environment must be rebuilt manually.

## Key commands
```
.venv_tf\Scripts\Activate                              # activate TF environment (Python 3.12)
.venv\Scripts\Activate                                 # activate PyTorch/sklearn environment (Python 3.14)
$env:TF_ENABLE_ONEDNN_OPTS="0"                         # set before running TF scripts to avoid oneDNN issues
python SHASHANK\scripts\convert_weights.py              # PyTorch → TF weight converter (portable)
python SHASHANK\scripts\train_tf.py                     # fine-tune converted TF model
python SHASHANK\scripts\webcam_predict.py               # real-time webcam inference
python SHASHANK\scripts\Mobilenet_train.py              # fine-tune TF model on actual dataset (fixes classifier)
python ANUSHRI\soil_predict.py                          # soil/crop prediction
python ANUSHRI\plant_predict.py                         # leaf disease inference using SHASHANK model
jupyter notebook ANUSHRI\soil_health_pipeline.ipynb     # soil model training
```

## Architecture quirks
- **Dual-framework vision pipeline:** Model is trained in PyTorch, then weights are manually mapped to a reimplemented TF architecture via `convert_weights.py`. Inference operates on the TF `.keras` model.
- **`Mobilenet_train.py` must be run after `convert_weights.py`** — it loads the converted model and fine-tunes the classifier head to work with the ImageNet-pretrained backbone. Without this step, the classifier is trained on random (PyTorch) backbone features but deployed on ImageNet features, producing garbage predictions.
- **Model file naming:**
  - `cnn_mobilenet_tf_unfitted.keras` — raw PyTorch→TF conversion (broken classifier, do not use for inference)
  - `cnn_mobilenet_tf_best.keras` — best checkpoint from fine-tuning
  - `cnn_mobilenet_tf_final.keras` — final fine-tuned model (use for inference)
- **Confidence threshold of 30%** added to all inference scripts — predictions below 30% show "Low_Confidence___No_leaf_detected".
- **Model files are gitignored** (`.pt`, `.pth`, `.keras`, `.h5`, `.weights`). Clone does not include trained models.
- **Duplicate source files:** `Mobilenet_train.py` and `Mobilenet_train.txt` are identical (both in `scripts/`). Both train a TF model from scratch.
- **Notebook outputs are committed** — `soil_health_pipeline.ipynb` is 2163 lines mostly of cell output.
- **`undersample_dataset.py` is documented in README but does not exist in the repo.**
- **Undersampling** caps overrepresented classes at 2000 images (removed images stored in `DATASET\Augmentated_Removed_Images\`).
- **SHASHANK/ is organized:** `scripts/`, `models/`, `notebooks/`, `docs/`, `assets/`.

## Tests
None. Validation is via notebook cells (`Test_model.ipynb`) or inline evaluation in training scripts.

## Conventions
- Jupyter notebooks (`.ipynb`) are the primary development interface
- `Dataset.md` and `Dataset_Analysis.md` describe dataset structure
- `Components.md` details planned IoT hardware (Raspberry Pi 5, LoRa, AWS IoT, InfluxDB, Grafana) — none implemented in code
- `SHASHANK\models.md` documents CNN architecture (note: recommends 256x256 input, but actual code uses 224x224)
