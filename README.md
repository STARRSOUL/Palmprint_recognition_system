# Deep Learning Contactless Palmprint Recognition

Lightweight ArcFace-based identity recognition for contactless palmprints.  
Notebook: [palmprint_cnn_fixed(2).ipynb](palmprint_cnn_fixed(2).ipynb)

## Overview
- Trains a CNN backbone that produces 512-d unit embeddings and an ArcFace head for identity supervision.
- Uses session1 images for training and session2 images for testing (one image per identity per session).
- Rank-1 1-NN on embeddings (session1 gallery → session2 probes) is the primary useful metric.

## Repo layout
- [palmprint_cnn_fixed(2).ipynb](palmprint_cnn_fixed(2).ipynb) — main notebook (all functions & cells).
- dataset/ — place your dataset here with `session1/` and `session2/`.
- outputs/ — saved checkpoints and visualizations (backbone.keras, full_model.keras, best_model.keras, training_curves.png, gradcam_xai.png, classification_report.txt).
- palmprint_model.h5 — optional saved model snapshot.

## Important functions & symbols
- Preprocessing & IO:
  - [`load_dataset`](palmprint_cnn_fixed(2).ipynb)
  - [`load_and_preprocess`](palmprint_cnn_fixed(2).ipynb)
  - [`extract_roi`](palmprint_cnn_fixed(2).ipynb)
- Model building & training:
  - [`build_backbone`](palmprint_cnn_fixed(2).ipynb)
  - [`build_model`](palmprint_cnn_fixed(2).ipynb)
  - [`train_model`](palmprint_cnn_fixed(2).ipynb)
  - [`ArcFaceLayer`](palmprint_cnn_fixed(2).ipynb)
- Evaluation / XAI:
  - [`evaluate_model`](palmprint_cnn_fixed(2).ipynb)
  - [`get_gradcam_heatmap`](palmprint_cnn_fixed(2).ipynb)
  - [`visualize_gradcam`](palmprint_cnn_fixed(2).ipynb)
- Files you will use:
  - [dataset/](dataset/)
  - [outputs/](outputs/)
  - [palmprint_model.h5](palmprint_model.h5)

## Dataset

DATASET_URL = https://www.kaggle.com/datasets/saqibshoaibdz/palm-dataset/data

- Prepare a folder `dataset/` with two subfolders: `session1/` (training) and `session2/` (testing). Filenames should be identity IDs (e.g. `00001.tiff`).
- The notebook expects one image per identity in each session (standard palmprint protocol). If you use any Other dataset, place files accordingly.

If you need public datasets, look for contactless palmprint datasets (e.g., PolyU contactless palmprint / authors' dataset pages). Place downloaded images into the structure above.

## Requirements
Install dependencies (example):
```sh
pip install -r requirements.txt
# or
pip install tensorflow opencv-python-headless matplotlib scikit-learn tqdm pillow psutil
```

## Quick start
1. Open [palmprint_cnn_fixed(2).ipynb](palmprint_cnn_fixed(2).ipynb) in Jupyter / VS Code.
2. Update configuration at the top (DATASET_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE).
3. Run cells in order:
   - Load dataset (`load_dataset`)
   - Visualize samples
   - Train: run the "Train Model" cell which calls [`train_model`](palmprint_cnn_fixed(2).ipynb)
   - Plot training curves (`plot_training_curves`)
   - Evaluate (`evaluate_model`) and Grad-CAM (`visualize_gradcam`)

## Training tips
- Do not rely on Keras "accuracy" for ArcFace when training with ~1 sample/class — it will be near-zero and misleading. Use the provided 1-NN Rank-1 callback (`Rank1Callback` inside [`train_model`](palmprint_cnn_fixed(2).ipynb)) which logs `val_rank1`.
- Recommended hyperparams in notebook: lr=1e-4, batch size >= 32 (increase if GPU memory permits), epochs 50–80.
- Use the `outputs/` best model (`best_model.keras`) and the backbone (`backbone.keras`) for embedding-based evaluation.

## How to load saved models
- Backbone (inference embeddings):
```python
import tensorflow as tf
backbone = tf.keras.models.load_model("outputs/backbone.keras", compile=False)
```
- Full ArcFace training model:
```python
model = tf.keras.models.load_model("outputs/full_model.keras", compile=False)
```

## Outputs produced
- outputs/backbone.keras — saved backbone model
- outputs/full_model.keras — saved training model
- outputs/best_model.keras — saved best checkpoint (monitored by Rank-1)
- outputs/training_curves.png — loss/accuracy plots (note: accuracy curve is not meaningful for ArcFace with 1 sample/class)
- outputs/gradcam_xai.png — Grad-CAM examples
- outputs/classification_report.txt — final classification report

## Notes & known pitfalls
- The notebook includes diagnostics to verify L2-normalization of embeddings. Keep that check.
- If training is slow: reduce `EVAL_LIMIT` inside `Rank1Callback` or reduce `batch_size` in the callback only.
- For reproducibility set `RANDOM_SEED`.

## Contact
- Issues / improvements: open an issue in this repository and reference [palmprint_cnn_fixed(2).ipynb](palmprint_cnn_fixed(2).ipynb).
