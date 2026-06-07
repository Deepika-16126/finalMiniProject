"""
Generate PCA scatter and pie chart images from handwriting_results.csv
Saves images into static/images/
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT = os.path.dirname(__file__)
CSV = os.path.join(ROOT, '..', 'handwriting_results.csv')
OUT_DIR = os.path.join(ROOT, 'static', 'images')
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CSV):
    print(f"CSV not found at {CSV} — skipping analytics image generation")
    raise SystemExit(1)

df = pd.read_csv(CSV)

FEATURE_NAMES = [
    "Spacing (CoV)",
    "Baseline Variation",
    "Fluency (px/stroke)",
    "Stroke Breaks",
    "Smoothness (roughness)",
]

X = df[FEATURE_NAMES].values.astype(float)
labels = df['cluster'].values if 'cluster' in df.columns else None

# PCA scatter
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

plt.figure(figsize=(7, 6))
colors = {0: '#639922', 1: '#378ADD', 2: '#D9534F'}
for cid in np.unique(labels):
    mask = labels == cid
    plt.scatter(X2[mask, 0], X2[mask, 1], label=f"Cluster {cid}", alpha=0.8, s=40, color=colors.get(cid, None))

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA Scatter — Handwriting (2D)')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
pca_path = os.path.join(OUT_DIR, 'pca_scatter.png')
plt.savefig(pca_path, dpi=150)
plt.close()
print(f"Saved PCA scatter → {pca_path}")

# Pie chart of labels
counts = pd.Series(labels).value_counts().sort_index()
labels_names = ['Good', 'Excellent', 'Bad']
plt.figure(figsize=(5, 5))
plt.pie(counts, labels=[labels_names[i] if i < len(labels_names) else str(i) for i in counts.index], autopct='%1.1f%%', colors=[colors.get(i) for i in counts.index])
plt.title('Quality distribution')
pie_path = os.path.join(OUT_DIR, 'pie_chart.png')
plt.savefig(pie_path, dpi=150)
plt.close()
print(f"Saved pie chart → {pie_path}")

# Feature comparison (mean per cluster)
if 'cluster' in df.columns:
    centroids = df.groupby('cluster')[FEATURE_NAMES].mean()
    plt.figure(figsize=(9, 5))
    centroids.plot(kind='bar')
    plt.title('Feature comparison — cluster means')
    plt.ylabel('Value')
    plt.xticks(rotation=0)
    plt.tight_layout()
    feat_path = os.path.join(OUT_DIR, 'feature_compare.png')
    plt.savefig(feat_path, dpi=150)
    plt.close()
    print(f"Saved feature comparison → {feat_path}")

# If classifier results image exists in project root, copy it to static/images
import shutil
possible = os.path.join(ROOT, '..', 'classifier_results.png')
if os.path.exists(possible):
    try:
        shutil.copy(possible, os.path.join(OUT_DIR, 'classifier_results.png'))
        print('Copied existing classifier_results.png to static/images')
    except Exception:
        pass
