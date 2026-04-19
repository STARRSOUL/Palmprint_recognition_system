""""
Deep Learning Based Contactless Palm Print Recognition
=======================================================
Based on the research paper by Shubh Agarwal, Anurag Parashar, Yash Gupta,
Vaibhav Vishwakarma, and Gurpreet Kour Khalsa — SRM Institute of Science & Technology.

DATASET STRUCTURE EXPECTED:
    archive/
        session1/
            00001.tiff   <- Person 1, session 1
            00002.tiff   <- Person 2, session 1
            ...
        session2/
            00001.tiff   <- Person 1, session 2
            00002.tiff   <- Person 2, session 2
            ...

HOW IT WORKS:
    - Each filename (e.g. 00001) = one unique person/class.
    - session1 images are used for TRAINING.
    - session2 images are used for TESTING (standard protocol for palmprint datasets).
    - This is how real biometric systems are evaluated.

INSTALL:
    pip install tensorflow opencv-python scikit-learn matplotlib tqdm pillow

RUN:
    python palmprint_cnn.py

PREDICT a single image after training:
    python palmprint_cnn.py predict path/to/palm.tiff

"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  <- Edit DATASET_DIR to point to your archive folder
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR   = "dataset"          # folder containing session1 and session2
SESSION_TRAIN = "session1"         # folder used for training
SESSION_TEST  = "session2"         # folder used for testing
IMG_SIZE      = (128, 128)         # resize target (H, W)
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 0.001
VAL_SPLIT     = 0.15               # fraction of training data used for validation
RANDOM_SEED   = 42
MODEL_SAVE    = "palmprint_model.h5"
OUTPUT_DIR    = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ─────────────────────────────────────────────────────────────────────────────
#  1. ROI EXTRACTION
#     Your images have a black background which makes extraction very easy.
# ─────────────────────────────────────────────────────────────────────────────

def extract_roi(image_bgr):
    """
    Automated ROI extraction optimised for black-background palmprint images:
      1. Grayscale + simple threshold (black bg makes this trivial)
      2. Morphological cleanup
      3. Find largest contour (the palm)
      4. Crop bounding box with padding
    Falls back to full image if detection fails.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) \
           if len(image_bgr.shape) == 3 else image_bgr.copy()

    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 15
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image_bgr.shape[1] - x, w + 2 * pad)
        h = min(image_bgr.shape[0] - y, h + 2 * pad)
        roi = image_bgr[y:y+h, x:x+w]
        if roi.size > 0:
            return roi

    return image_bgr   # fallback: use whole image


def load_and_preprocess(filepath):
    """
    Full preprocessing pipeline per image:
      1. Read (PIL fallback for TIFF)
      2. ROI extraction
      3. Resize to IMG_SIZE
      4. Normalize to [0, 1]
    """
    img = cv2.imread(str(filepath))

    if img is None:
        try:
            from PIL import Image
            pil_img = Image.open(str(filepath)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    roi = extract_roi(img)
    roi = cv2.resize(roi, (IMG_SIZE[1], IMG_SIZE[0]))   # cv2 uses (W, H)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi.astype(np.float32) / 255.0
    return roi


# ─────────────────────────────────────────────────────────────────────────────
#  2. DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_session(session_path, person_to_label):
    """Load all images from one session folder."""
    images, labels = [], []
    files = sorted([f for f in session_path.iterdir()
                    if f.suffix.lower() in SUPPORTED_EXTS])

    for fpath in tqdm(files, desc=f"  {session_path.name}", leave=False):
        person_id = fpath.stem
        if person_id not in person_to_label:
            continue
        img = load_and_preprocess(fpath)
        if img is not None:
            images.append(img)
            labels.append(person_to_label[person_id])

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def load_dataset(dataset_dir):
    """
    Build class map from session1 filenames, then load both sessions.
    Returns X_train, y_train, X_test, y_test, class_names.
    """
    base    = Path(dataset_dir)
    train_p = base / SESSION_TRAIN
    test_p  = base / SESSION_TEST

    if not train_p.exists():
        sys.exit(f"[ERROR] Training folder not found: {train_p}")
    if not test_p.exists():
        sys.exit(f"[ERROR] Test folder not found: {test_p}")

    train_files     = sorted([f for f in train_p.iterdir()
                               if f.suffix.lower() in SUPPORTED_EXTS])
    person_ids      = sorted(set(f.stem for f in train_files))
    person_to_label = {pid: idx for idx, pid in enumerate(person_ids)}
    class_names     = person_ids

    print(f"\n[INFO] Unique persons detected : {len(class_names)}")
    print(f"[INFO] Training session        : {SESSION_TRAIN}")
    print(f"[INFO] Test session            : {SESSION_TEST}")

    print("\n[INFO] Loading training data...")
    X_train, y_train = load_session(train_p, person_to_label)
    print("[INFO] Loading test data...")
    X_test,  y_test  = load_session(test_p,  person_to_label)

    print(f"\n[INFO] Train: {len(X_train)} images | Test: {len(X_test)} images")
    print(f"[INFO] Image shape: {X_train[0].shape}")

    return X_train, y_train, X_test, y_test, class_names


# ─────────────────────────────────────────────────────────────────────────────
#  3. DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def get_augmentation_layer():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.08),
        layers.RandomBrightness(0.08),
        layers.RandomContrast(0.08),
    ], name="augmentation")


# ─────────────────────────────────────────────────────────────────────────────
#  4. CNN MODEL ARCHITECTURE
#     Conv blocks -> ReLU -> MaxPool -> Flatten -> FC -> Softmax
#     Matches the architecture described in the paper.
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_palm_image")
    x = get_augmentation_layer()(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 5 (extra depth helps with many persons)
    x = layers.Conv2D(512, (3, 3), padding="same", name="conv5")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Fully Connected
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Softmax output -> confidence score per person
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return models.Model(inputs, outputs, name="PalmPrint_CNN")


# ─────────────────────────────────────────────────────────────────────────────
#  5. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, num_classes):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
       # stratify=y_train
    )

    model = build_model(num_classes)
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=8,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, verbose=1, min_lr=1e-6
        ),
        callbacks.ModelCheckpoint(
            MODEL_SAVE, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    y_tr_cat  = to_categorical(y_tr,  num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    print(f"\n[INFO] Training on {len(X_tr)} | Validating on {len(X_val)}")
    history = model.fit(
        X_tr, y_tr_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1
    )
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
#  6. EVALUATION & PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def top_k_accuracy(y_true, y_pred_probs, k=5):
    top_k   = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k[i] for i in range(len(y_true)))
    return correct / len(y_true)


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"],     label="Train Loss",     linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss",       linewidth=2, linestyle="--")
    axes[0].set_title("Training Loss vs Epochs", fontsize=14)
    axes[0].set_xlabel("Epochs"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy",   linewidth=2, linestyle="--")
    axes[1].set_title("Training Accuracy vs Epochs", fontsize=14)
    axes[1].set_xlabel("Epochs"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Training curves saved -> {path}")


def evaluate_model(model, X_test, y_test, class_names):
    num_classes  = len(class_names)
    y_test_cat   = to_categorical(y_test, num_classes)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    print(f"\n{'='*55}")
    print(f"  Test Accuracy  : {test_acc * 100:.2f}%")
    print(f"  Test Loss      : {test_loss:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    top5 = top_k_accuracy(y_test, y_pred_probs, k=5)
    print(f"  Top-5 Accuracy : {top5 * 100:.2f}%")
    print(f"{'='*55}")

    if num_classes <= 50:
        report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    else:
        report = classification_report(y_test, y_pred, digits=4)

    print("\n[Classification Report]\n")
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy : {test_acc * 100:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5 * 100:.2f}%\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n\n")
        f.write(report)
    print(f"[INFO] Report saved -> {report_path}")

    if num_classes <= 50:
        cm     = confusion_matrix(y_test, y_pred)
        fsz    = max(8, num_classes // 3)
        fig, ax = plt.subplots(figsize=(fsz, fsz))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, cmap="Blues", colorbar=False, xticks_rotation="vertical"
        )
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150); plt.close()
        print(f"[INFO] Confusion matrix saved -> {cm_path}")
    else:
        print(f"[INFO] Skipping confusion matrix (>{50} classes)")

    return y_pred, y_pred_probs


# ─────────────────────────────────────────────────────────────────────────────
#  7. EXPLAINABLE AI — Grad-CAM
#     Shows which palm regions the CNN focused on for each prediction.
# ─────────────────────────────────────────────────────────────────────────────

def get_gradcam_heatmap(model, img_array, last_conv="conv5"):
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        pred_cls        = tf.argmax(preds[0])
        score           = preds[:, pred_cls]

    grads   = tape.gradient(score, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = (conv_out[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()
    return heatmap, int(pred_cls), float(tf.reduce_max(preds).numpy())


def visualize_gradcam(model, X_test, y_test, class_names, num_samples=6):
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3.5))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img       = X_test[idx]
        heatmap, pred_cls, conf = get_gradcam_heatmap(model, np.expand_dims(img, 0))

        hmap    = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
        colored = cv2.applyColorMap((hmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted((img * 255).astype(np.uint8), 0.6, colored, 0.4, 0)

        true_lbl = class_names[y_test[idx]]
        pred_lbl = class_names[pred_cls]

        axes[row, 0].imshow(img);         axes[row, 0].set_title("Input Palm");       axes[row, 0].axis("off")
        axes[row, 1].imshow(hmap, cmap="jet"); axes[row, 1].set_title("Grad-CAM (XAI)"); axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(
            f"Pred: {pred_lbl} ({conf*100:.1f}%)\nTrue: {true_lbl}",
            color="green" if pred_lbl == true_lbl else "red"
        )
        axes[row, 2].axis("off")

    plt.suptitle("Explainable AI (Grad-CAM) — Palm Region Importance", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gradcam_xai.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[INFO] Grad-CAM XAI saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  8. SAMPLE VISUALISATION — sanity check before training starts
# ─────────────────────────────────────────────────────────────────────────────

def visualize_samples(X_train, y_train, class_names, n=8):
    n       = min(n, len(X_train))
    indices = np.random.choice(len(X_train), n, replace=False)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 3))
    for ax, idx in zip(axes, indices):
        ax.imshow(X_train[idx])
        ax.set_title(f"ID: {class_names[y_train[idx]]}", fontsize=7)
        ax.axis("off")
    plt.suptitle("Sample Training Images (after ROI + Preprocessing)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sample_images.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"[INFO] Sample images saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  9. SINGLE IMAGE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(model, image_path, class_names):
    img = load_and_preprocess(image_path)
    if img is None:
        print(f"[ERROR] Could not load: {image_path}"); return

    probs     = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    pred_cls  = int(np.argmax(probs))
    conf      = float(probs[pred_cls])
    top5_idx  = np.argsort(probs)[::-1][:5]

    print(f"\n[Prediction for: {image_path}]")
    print(f"  Best Match : Person {class_names[pred_cls]}  ({conf*100:.1f}% confidence)")
    print(f"\n  Top-5 Matches:")
    for rank, i in enumerate(top5_idx, 1):
        print(f"    {rank}. Person {class_names[i]:>8}  ->  {probs[i]*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
#  10. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Deep Learning Contactless PalmPrint Recognition")
    print("  Mode: Person Identity Recognition")
    print("="*60)

    X_train, y_train, X_test, y_test, class_names = load_dataset(DATASET_DIR)
    num_classes = len(class_names)

    visualize_samples(X_train, y_train, class_names)
    model, history = train_model(X_train, y_train, num_classes)
    plot_training_curves(history)
    evaluate_model(model, X_test, y_test, class_names)
    visualize_gradcam(model, X_test, y_test, class_names)

    print(f"\n[INFO] Model saved  -> {MODEL_SAVE}")
    print(f"[INFO] All outputs  -> {OUTPUT_DIR}/")
    print("\nPipeline complete!")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "predict":
        if not os.path.exists(MODEL_SAVE):
            sys.exit(f"[ERROR] No model found at {MODEL_SAVE}. Train first.")
        m = models.load_model(MODEL_SAVE)
        base   = Path(DATASET_DIR) / SESSION_TRAIN
        cnames = sorted(set(f.stem for f in base.iterdir()
                            if f.suffix.lower() in SUPPORTED_EXTS))
        predict_single(m, sys.argv[2], cnames)
    else:
        main()

