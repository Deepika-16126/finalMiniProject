"""
=============================================================
  HANDWRITING QUALITY — SUPERVISED CLASSIFIER
  Adapted for local execution (uses local CSV `handwriting_results.csv`)
=============================================================
"""

# Remove Google Colab mount and force non-interactive matplotlib backend
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".cache_matplotlib"))

import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────
# STEP 1 ▸ Imports
# ─────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib
import json

from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.svm               import SVC
from sklearn.ensemble          import RandomForestClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       confusion_matrix,
                                       classification_report)


# ─────────────────────────────────────────────
# STEP 2 ▸ Configuration (LOCAL PATHS)
# ─────────────────────────────────────────────
CSV_PATH     = "handwriting_results_chgd.csv"
DATASET_PATH = "handwriting_400"
IMG_SIZE     = (128, 128)
RANDOM_STATE = 42
TEST_SIZE    = 0.2    # 80% train, 20% test

FEATURE_NAMES = [
    "Spacing (CoV)",
    "Baseline Variation",
    "Fluency (px/stroke)",
    "Stroke Breaks",
    "Smoothness (roughness)",
]

CLASS_NAMES = ["Bad", "Good", "Excellent"]


# ─────────────────────────────────────────────
# STEP 3 ▸ Load CSV — features + labels
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 3 — Loading CSV …")
print("=" * 60)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH} — place `handwriting_results.csv` in project root")


df = pd.read_csv(CSV_PATH)

print(f"\n  ✓ Loaded {len(df)} rows from CSV")
print(f"  Columns : {list(df.columns)}")

# ── Map cluster IDs to quality labels ────────
CLUSTER_LABEL_MAP = {
    0: "Good",
    1: "Excellent",
    2: "Bad",
}

# Create quality_label column from cluster column
if 'cluster' not in df.columns:
    raise ValueError("CSV missing required 'cluster' column")

df['quality_label'] = df['cluster'].map(CLUSTER_LABEL_MAP)

print(f"\n  Cluster → Label mapping applied:")
for cid, label in CLUSTER_LABEL_MAP.items():
    count = int((df['cluster'] == cid).sum())
    print(f"    Cluster {cid} → {label:<12s}: {count} images")

print(f"\n  Label distribution:")
for label, count in df['quality_label'].value_counts().items():
    print(f"    {label:<12s}: {count} images  "
          f"({count/len(df)*100:.1f}%)")
print()

# Features matrix X — using exact column names from CSV
feature_cols = FEATURE_NAMES
print(f"  Feature columns used: {feature_cols}")
X = df[feature_cols].values.astype(np.float64)
y = df['quality_label'].values

print(f"  X shape : {X.shape}  "
      f"({X.shape[0]} images × {X.shape[1]} features)")
print(f"  y shape : {y.shape}\n")


# ─────────────────────────────────────────────
# STEP 4 ▸ Encode Labels
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 4 — Encoding labels …")
print("=" * 60)

le = LabelEncoder()
le.fit(CLASS_NAMES)          # fix order: Bad=0, Good=1, Excellent=2
y_encoded = le.transform(y)

print(f"\n  Label encoding:")
for cls, idx in zip(le.classes_, range(len(le.classes_))):
    print(f"    {cls:<12s} → {idx}")
print()


# ─────────────────────────────────────────────
# STEP 5 ▸ Train / Test Split (80 / 20)
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 5 — Train/Test Split (80/20) …")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y_encoded
)

print(f"\n  Training set : {X_train.shape[0]} images  (80%)")
print(f"  Test set     : {X_test.shape[0]}  images  (20%)")
print(f"\n  Train class distribution:")
for idx, name in enumerate(le.classes_):
    count = int(np.sum(y_train == idx))
    print(f"    {name:<12s}: {count}")
print(f"\n  Test class distribution:")
for idx, name in enumerate(le.classes_):
    count = int(np.sum(y_test == idx))
    print(f"    {name:<12s}: {count}")
print()


# ─────────────────────────────────────────────
# STEP 6 ▸ Scale Features
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 6 — Scaling features …")
print("=" * 60)

scaler   = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("  ✓ StandardScaler fitted on training data")
print("  ✓ Test data transformed using training scaler\n")


# ─────────────────────────────────────────────
# STEP 7 ▸ Train 3 Classifiers
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 7 — Training classifiers …")
print("=" * 60)

classifiers = {
    "SVM": SVC(
        kernel      = 'rbf',
        C           = 10,
        gamma       = 'scale',
        probability = True,
        random_state= RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators = 200,
        max_depth    = None,
        class_weight = 'balanced',
        random_state = RANDOM_STATE
    ),
    "Logistic Regression": LogisticRegression(
        max_iter     = 1000,
        class_weight = 'balanced',
        random_state = RANDOM_STATE
    ),
}

trained_models = {}
results        = {}

for name, clf in classifiers.items():
    print(f"\n  Training {name} …")
    clf.fit(X_train_scaled, y_train)
    trained_models[name] = clf

    y_pred = clf.predict(X_test_scaled)

    acc            = accuracy_score(y_test, y_pred)
    prec_weighted  = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted   = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted    = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec_macro     = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro      = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro       = f1_score(y_test, y_pred, average='macro', zero_division=0)

    cv_scores = cross_val_score(clf, scaler.transform(X), y_encoded, cv=5, scoring='f1_weighted')

    results[name] = {
        'accuracy'           : acc,
        'precision'          : prec_weighted,
        'recall'             : rec_weighted,
        'f1'                 : f1_weighted,
        'precision_weighted' : prec_weighted,
        'recall_weighted'    : rec_weighted,
        'f1_weighted'        : f1_weighted,
        'precision_macro'    : prec_macro,
        'recall_macro'       : rec_macro,
        'f1_macro'           : f1_macro,
        'cv_mean'            : cv_scores.mean(),
        'cv_std'             : cv_scores.std(),
        'y_pred'             : y_pred,
    }

    print(f"    Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"    Precision : {prec_weighted:.4f} weighted | {prec_macro:.4f} macro")
    print(f"    Recall    : {rec_weighted:.4f} weighted | {rec_macro:.4f} macro")
    print(f"    F1 Score  : {f1_weighted:.4f} weighted | {f1_macro:.4f} macro")
    print(f"    CV F1     : {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

print()


# ─────────────────────────────────────────────
# STEP 8 ▸ Terminal Evaluation Summary
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 8 — Terminal Evaluation Summary")
print("=" * 60)

print("\n  Overall metrics on test set")
print("  " + "-" * 103)
print(
    "  "
    f"{'Model':<22s}"
    f"{'Accuracy':>10s}"
    f"{'Prec(W)':>10s}"
    f"{'Recall(W)':>11s}"
    f"{'F1(W)':>9s}"
    f"{'Prec(M)':>10s}"
    f"{'Recall(M)':>11s}"
    f"{'F1(M)':>9s}"
    f"{'CV F1':>11s}"
)
print("  " + "-" * 103)
for name, res in results.items():
    print(
        "  "
        f"{name:<22s}"
        f"{res['accuracy']:>10.4f}"
        f"{res['precision_weighted']:>10.4f}"
        f"{res['recall_weighted']:>11.4f}"
        f"{res['f1_weighted']:>9.4f}"
        f"{res['precision_macro']:>10.4f}"
        f"{res['recall_macro']:>11.4f}"
        f"{res['f1_macro']:>9.4f}"
        f"{res['cv_mean']:>8.4f} ± {res['cv_std']:.4f}"
    )
print("  " + "-" * 103)
print("  W = weighted average, M = macro average\n")

print("  Detailed classification reports and confusion matrices")
for name, res in results.items():
    print(f"\n  ── {name} ──")
    print(classification_report(y_test, res['y_pred'], target_names = le.classes_, zero_division= 0))
    cm = confusion_matrix(y_test, res['y_pred'])
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {cls}" for cls in le.classes_],
        columns=[f"Pred {cls}" for cls in le.classes_],
    )
    print("  Confusion Matrix:")
    print(cm_df.to_string())
    print()


# ─────────────────────────────────────────────
# STEP 9 ▸ Visualisations
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 9 — Generating visualisations …")
print("=" * 60)

fig = plt.figure(figsize=(18, 14), facecolor="white")
fig.suptitle("Handwriting Quality — Supervised Classifier Results", fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

clf_names = list(results.keys())
colors    = ["#378ADD", "#639922", "#7F77DD"]

# Row 1: Confusion matrices
for col, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[0, col])
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues', ax=ax, cbar=False, linewidths=0.5)
    ax.set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.tick_params(labelsize=8)

# Row 2 Left+Mid: Metric comparison bar chart
ax_metrics = fig.add_subplot(gs[1, :2])
metrics     = ['accuracy', 'precision', 'recall', 'f1']
x           = np.arange(len(metrics))
bar_w       = 0.25

for i, (name, res) in enumerate(results.items()):
    vals   = [res[m] for m in metrics]
    offset = (i - 1) * bar_w
    bars   = ax_metrics.bar(x + offset, vals, bar_w, label=name, color=colors[i], alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax_metrics.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=7, fontweight='bold')

ax_metrics.set_xticks(x)
ax_metrics.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'], fontsize=10)
ax_metrics.set_ylim(0, 1.15)
ax_metrics.set_ylabel("Score", fontsize=10)
ax_metrics.set_title("Metric Comparison — All 3 Classifiers", fontsize=11, fontweight="bold")
ax_metrics.legend(fontsize=9)
ax_metrics.grid(True, axis='y', alpha=0.3)
ax_metrics.spines['top'].set_visible(False)
ax_metrics.spines['right'].set_visible(False)

# Row 2 Right: CV F1 comparison
ax_cv = fig.add_subplot(gs[1, 2])
cv_means = [results[n]['cv_mean'] for n in clf_names]
cv_stds  = [results[n]['cv_std']  for n in clf_names]
bars = ax_cv.bar(clf_names, cv_means, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
ax_cv.errorbar(clf_names, cv_means, yerr=cv_stds, fmt='none', color='black', capsize=5, linewidth=1.5)
for bar, val in zip(bars, cv_means):
    ax_cv.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax_cv.set_ylim(0, 1.2)
ax_cv.set_ylabel("CV F1 Score", fontsize=10)
ax_cv.set_title("5-Fold Cross Validation\nF1 Score", fontsize=11, fontweight="bold")
ax_cv.tick_params(axis='x', labelsize=8)
ax_cv.grid(True, axis='y', alpha=0.3)
ax_cv.spines['top'].set_visible(False)
ax_cv.spines['right'].set_visible(False)

# Row 3: Feature importance (Random Forest)
ax_fi = fig.add_subplot(gs[2, :2])
rf_model    = trained_models["Random Forest"]
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

bars = ax_fi.bar(range(len(FEATURE_NAMES)), importances[sorted_idx], color="#639922", alpha=0.85, edgecolor="white", linewidth=0.5)
ax_fi.set_xticks(range(len(FEATURE_NAMES)))
ax_fi.set_xticklabels([FEATURE_NAMES[i] for i in sorted_idx], rotation=15, ha='right', fontsize=9)
for bar, val in zip(bars, importances[sorted_idx]):
    ax_fi.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
ax_fi.set_ylabel("Importance", fontsize=10)
ax_fi.set_title("Feature Importance — Random Forest", fontsize=11, fontweight="bold")
ax_fi.grid(True, axis='y', alpha=0.3)
ax_fi.spines['top'].set_visible(False)
ax_fi.spines['right'].set_visible(False)

# Row 3 Right: Summary table
ax_tbl = fig.add_subplot(gs[2, 2])
ax_tbl.axis('off')

table_data = []
for name in clf_names:
    r = results[name]
    table_data.append([name.replace(" ", "\n"), f"{r['accuracy']*100:.1f}%", f"{r['f1']:.3f}", f"{r['cv_mean']:.3f}"])

tbl = ax_tbl.table(cellText = table_data, colLabels = ['Model', 'Accuracy', 'F1', 'CV F1'], loc = 'center', cellLoc = 'center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 2.2)

# Style header
for j in range(4):
    tbl[0, j].set_facecolor('#1a1a2e')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Highlight best F1 row
best_name = max(results, key=lambda n: results[n]['f1'])
best_row  = clf_names.index(best_name) + 1
for j in range(4):
    tbl[best_row, j].set_facecolor('#EAF3DE')
    tbl[best_row, j].set_text_props(color='#27500A', fontweight='bold')

ax_tbl.set_title("Summary — Best model\nhighlighted in green", fontsize=10, fontweight="bold", pad=10)

# ensure target static image folder exists (used by the Flask app)
STATIC_IMG_DIR = os.path.join(DATASET_PATH, 'static', 'images')
os.makedirs(STATIC_IMG_DIR, exist_ok=True)
out_img_path = os.path.join(STATIC_IMG_DIR, "classifier_results.png")
plt.savefig(out_img_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  ✓ Saved → {out_img_path}\n")


# ─────────────────────────────────────────────
# STEP 10 ▸ Best Model Summary
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 10 — Best Model")
print("=" * 60)

best_model_name = max(results, key=lambda n: results[n]['f1'])
best_model      = trained_models[best_model_name]
best_res        = results[best_model_name]

print(f"\n  Best classifier : {best_model_name}")
print(f"  Accuracy        : {best_res['accuracy']*100:.2f}%")
print(f"  Precision       : {best_res['precision']:.4f}")
print(f"  Recall          : {best_res['recall']:.4f}")
print(f"  F1 Score        : {best_res['f1']:.4f}")
print(f"  CV F1 (5-fold)  : {best_res['cv_mean']:.4f} (± {best_res['cv_std']:.4f})\n")

# ─────────────────────────────────────────────
# Save best model + scaler + label encoder for the Flask app
# ─────────────────────────────────────────────
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(models_dir, 'best_model.joblib'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
joblib.dump(le, os.path.join(models_dir, 'labelencoder.joblib'))

# create a JSON-safe copy of results (exclude raw arrays like y_pred)
serial_results = {}
for mname, r in results.items():
    serial_results[mname] = {
        'accuracy': float(r.get('accuracy', 0.0)),
        'precision': float(r.get('precision', 0.0)),
        'recall': float(r.get('recall', 0.0)),
        'f1': float(r.get('f1', 0.0)),
        'cv_mean': float(r.get('cv_mean', 0.0)),
        'cv_std': float(r.get('cv_std', 0.0))
    }

best_metrics = serial_results.get(best_model_name, {})

metrics = {
    'best_model': best_model_name,
    'best_metrics': best_metrics,
    'all_results': serial_results
}
with open(os.path.join(models_dir, 'metrics.json'), 'w') as fh:
    json.dump(metrics, fh, indent=2)

print(f"  ✓ Saved trained model artifacts → {models_dir}\n")

# ─────────────────────────────────────────────
# STEP 11 ▸ Predict New Image
# ─────────────────────────────────────────────

# Preprocessing and feature extraction (same functions as KMeans pipeline)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    
    if img is None:
        return None
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    denoised= cv2.GaussianBlur(resized, (3, 3), 0)
    binary  = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=4)
    return binary

# feature functions (spacing, baseline, fluency, stroke breaks, smoothness)

def feature_spacing(binary):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels < 3:
        return 0.0
    x_centres = sorted([stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 5])
    if len(x_centres) < 2:
        return 0.0
    gaps     = np.diff(x_centres)
    mean_gap = np.mean(gaps)
    return float(np.std(gaps) / mean_gap) if mean_gap != 0 else 0.0

def feature_baseline(binary):
    row_sums = binary.sum(axis=1).astype(float)
    return float(np.std(row_sums) / binary.shape[1])

def feature_fluency(binary):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    valid     = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 5]
    total_ink = float(np.sum(binary > 127))
    return float(total_ink / len(valid)) if valid else 0.0

def feature_stroke_breaks(binary):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    valid     = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 5]
    total_ink = float(np.sum(binary > 127))
    return float(len(valid) / (total_ink + 1e-6) * 1000) if total_ink > 0 else 0.0

def feature_smoothness(binary):
    lap = cv2.Laplacian(binary, cv2.CV_64F)
    return float(np.mean(np.abs(lap)))

def extract_features(binary):
    return [
        feature_spacing(binary),
        feature_baseline(binary),
        feature_fluency(binary),
        feature_stroke_breaks(binary),
        feature_smoothness(binary),
    ]


def predict_new_image_supervised(img_path):
    print("\n" + "=" * 55)
    print(f"  PREDICTING: {os.path.basename(img_path)}")
    print("=" * 55)

    binary = preprocess_image(img_path)
    if binary is None:
        print("  ERROR: Could not read image.")
        return None

    features     = extract_features(binary)
    features_arr = np.array(features, dtype=np.float64).reshape(1, -1)
    features_sc  = scaler.transform(features_arr)

    pred_encoded = best_model.predict(features_sc)[0]
    pred_label   = le.inverse_transform([pred_encoded])[0]

    probs = best_model.predict_proba(features_sc)[0]

    print(f"\n  Model     : {best_model_name}")
    print(f"  Prediction: {pred_label}")
    print(f"\n  Confidence:")
    for cls, prob in zip(le.classes_, probs):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<12s}: {prob*100:5.1f}%  {bar}")

    print(f"\n  Features extracted:")
    for name, val in zip(FEATURE_NAMES, features):
        print(f"    {name:<30s}: {val:.4f}")

    # Don't call plt.show() in non-interactive environment; display saved image instead
    raw      = cv2.imread(img_path)
    gray_disp= cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    h, w     = gray_disp.shape
    if max(h, w) > 1200:
        scale    = 1200 / max(h, w)
        gray_disp= cv2.resize(gray_disp, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    fig.suptitle(f"Prediction: {pred_label}  (Confidence: {max(probs)*100:.1f}%)", fontsize=13, fontweight="bold", color="#111")

    axes[0].imshow(gray_disp, cmap="gray")
    axes[0].set_title("Input Image", fontsize=11)
    axes[0].axis("off")

    colors_prob = ["#E24B4A", "#378ADD", "#639922"]
    axes[1].barh(le.classes_, probs, color=colors_prob, alpha=0.85, edgecolor="white")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability", fontsize=10)
    axes[1].set_title("Prediction Confidence", fontsize=11)
    for i, (prob, cls) in enumerate(zip(probs, le.classes_)):
        axes[1].text(prob + 0.01, i, f"{prob*100:.1f}%", va='center', fontsize=10, fontweight='bold')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    # Save the prediction figure to a file instead of showing
    outfn = f"prediction_{os.path.basename(img_path)}.png"
    plt.savefig(outfn, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved prediction visualization → {outfn}")

    return pred_label, probs


# ── HOW TO USE ────────────────────────────────
# To predict a new image after running this script, call:
# predict_new_image_supervised('/path/to/image.jpg')
# ─────────────────────────────────────────────

print("  ✅ Supervised classifier pipeline complete!")
print("  ─────────────────────────────────────────────────────")
print("  Outputs:")
print("    • classifier_results.png  — all metrics + confusion matrix")
print(f"    • Best model              — {best_model_name}")
print(f"    • Best F1 Score           — {best_res['f1']:.4f}")
print()
print("  To predict a new image:")
print("    predict_new_image_supervised('/path/to/image.jpg')")
print("  ─────────────────────────────────────────────────────\n")
