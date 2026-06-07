from flask import Flask, render_template, request, redirect, url_for
import os
import json
import uuid
import numpy as np
import pandas as pd
import cv2
import joblib
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(ROOT, '..')
CSV_PATH = os.path.join(PROJECT_ROOT, 'handwriting_results_chgd.csv')
UPLOAD_FOLDER = os.path.join(ROOT, 'static', 'uploads')
DATA_JSON = os.path.join(ROOT, 'data', 'analytics.json')
STATIC_IMG = os.path.join(ROOT, 'static', 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMG, exist_ok=True)

FEATURE_NAMES = [
    "Spacing (CoV)",
    "Baseline Variation",
    "Fluency (px/stroke)",
    "Stroke Breaks",
    "Smoothness (roughness)",
]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_dataset():
    if not os.path.exists(CSV_PATH):
        return None, None, None
    df = pd.read_csv(CSV_PATH)
    X = df[FEATURE_NAMES].values.astype(float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    # compute cluster centroids
    centroids = df.groupby('cluster')[FEATURE_NAMES].mean()
    centroids_scaled = scaler.transform(centroids.values)
    return df, scaler, (centroids.index.values, centroids_scaled, centroids)


def load_kmeans_models():
    """Load saved KMeans and scaler if available in models/"""
    kmeans_path = os.path.join(PROJECT_ROOT, 'models', 'kmeans.joblib')
    scaler_path = os.path.join(PROJECT_ROOT, 'models', 'kmeans_scaler.joblib')
    if os.path.exists(kmeans_path) and os.path.exists(scaler_path):
        try:
            kmeans = joblib.load(kmeans_path)
            kmeans_scaler = joblib.load(scaler_path)
            return kmeans, kmeans_scaler
        except Exception:
            return None, None
    return None, None


def load_supervised_model():
    """Load trained supervised classifier, scaler and label encoder if available."""
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.joblib')
    scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.joblib')
    le_path = os.path.join(PROJECT_ROOT, 'models', 'labelencoder.joblib')
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path):
        try:
            clf = joblib.load(model_path)
            clf_scaler = joblib.load(scaler_path)
            clf_le = joblib.load(le_path)
            return clf, clf_scaler, clf_le
        except Exception:
            return None, None, None
    return None, None, None


def save_history_item(item):
    try:
        with open(DATA_JSON, 'r') as f:
            history = json.load(f)
    except Exception:
        history = []
    history.insert(0, item)
    history = history[:5]
    with open(DATA_JSON, 'w') as f:
        json.dump(history, f, indent=2)


def read_history():
    try:
        with open(DATA_JSON, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    return binary


def extract_features(path):
    binary = preprocess_image(path)
    if binary is None:
        return None
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 5]
    widths = [stats[i, cv2.CC_STAT_WIDTH] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 5]
    # Spacing CoV — coefficient of variation of component widths
    spacing = 0.0
    if len(widths) >= 2:
        spacing = (np.std(widths) / (np.mean(widths) + 1e-8))
    # Baseline variation — std of y-centroids
    ys = [centroids[i][1] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 5]
    baseline_var = float(np.std(ys)) if len(ys) >= 2 else 0.0
    # Fluency — white pixels per stroke
    total_white = int(np.sum(binary > 0))
    strokes = max(1, len(areas))
    fluency = total_white / strokes
    # Stroke breaks — number of small components
    stroke_breaks = float(len(areas))
    # Smoothness — variance of Laplacian as roughness
    lap = cv2.Laplacian(binary, cv2.CV_64F)
    smoothness = float(np.var(lap))

    feat = {
        FEATURE_NAMES[0]: float(spacing),
        FEATURE_NAMES[1]: float(baseline_var),
        FEATURE_NAMES[2]: float(fluency),
        FEATURE_NAMES[3]: float(stroke_breaks),
        FEATURE_NAMES[4]: float(smoothness),
    }
    return feat


@app.route('/', methods=['GET', 'POST'])
def index():
    history = read_history()
    if request.method == 'POST':
        f = request.files.get('file')
        if not f:
            return redirect(url_for('index'))
        filename = f"{uuid.uuid4().hex}.png"
        dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(dest)
        return redirect(url_for('result', filename=filename))
    return render_template('index.html', history=history)


@app.route('/result')
def result():
    fname = request.args.get('filename')
    if not fname:
        return redirect(url_for('index'))
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    if not os.path.exists(path):
        return redirect(url_for('index'))

    # extract features and predict by KMeans model if available, else centroid fallback
    df, scaler, centroid_info = load_dataset()
    kmeans_model, kmeans_scaler = load_kmeans_models()
    features = extract_features(path)
    if features is None or df is None:
        # fallback: return a basic result
        item = {
            'id': uuid.uuid4().hex,
            'filename': fname,
            'cluster_id': None,
            'label': 'Unknown',
            'confidence': 0.0,
            'features': features,
            'distances': [],
            'reasons': []
        }
        save_history_item(item)
        return render_template('result.html', item=item)

    x = np.array([features[n] for n in FEATURE_NAMES], dtype=float).reshape(1, -1)
    X = df[FEATURE_NAMES].values.astype(float)
    sc = scaler
    x_s = sc.transform(x)

    # load supervised model if available
    sup_model, sup_scaler, sup_le = load_supervised_model()

    if kmeans_model is not None and kmeans_scaler is not None:
        # use kmeans scaler + model for prediction
        x_k = kmeans_scaler.transform(x)
        cluster_idx = int(kmeans_model.predict(x_k)[0])
        # distances to cluster centers (in scaled space)
        dists = kmeans_model.transform(x_k).flatten()
        # confidence as inverse normalized distance
        maxd = float(dists.max()) + 1e-6
        confidence = float((1.0 - (dists.min() / maxd)) * 100.0)
        best = cluster_idx
        centroid_vec = kmeans_model.cluster_centers_[best]
        # compare x_k to centroid for reasons
        diff = (x_k.flatten() - centroid_vec)
    else:
        # fallback: nearest centroid from CSV (older behavior)
        idxs, centroids_scaled, centroids_raw = centroid_info
        dists = np.linalg.norm(centroids_scaled - x_s, axis=1)
        best = int(idxs[np.argmin(dists)])
        maxd = float(dists.max()) + 1e-6
        confidence = float((1.0 - (dists.min() / maxd)) * 100.0)
        centroid_vec = centroids_scaled[list(idxs).index(best)]
        diff = (x_s.flatten() - centroid_vec)

    labels = {0: 'Good', 1: 'Excellent', 2: 'Bad'}
    label = labels.get(int(best), str(best))

    absdiff = np.abs(diff)
    top_idx = np.argsort(absdiff)[::-1][:3]
    reasons = []
    for i in top_idx:
        feat = FEATURE_NAMES[i]
        val = float(x.flatten()[i])
        direction = 'Higher than average' if diff[i] > 0 else 'Lower than average'
        impact = 'worse' if (feat in ['Stroke Breaks', 'Smoothness (roughness)', 'Baseline Variation', 'Spacing (CoV)'] and diff[i] > 0) else 'better'
        reasons.append({'feature': feat, 'value': val, 'direction': direction, 'impact': impact})

    # supervised prediction (if available)
    supervised = None
    if sup_model is not None and sup_scaler is not None and sup_le is not None:
        try:
            x_sup = sup_scaler.transform(x)
            sup_pred_idx = int(sup_model.predict(x_sup)[0])
            sup_label = sup_le.inverse_transform([sup_pred_idx])[0]
            sup_conf = 0.0
            sup_proba = None
            if hasattr(sup_model, 'predict_proba'):
                proba = sup_model.predict_proba(x_sup)[0]
                sup_proba = [float(p) for p in proba.tolist()]
                sup_conf = float(max(proba) * 100.0)
            else:
                sup_conf = float(0.0)

            # global feature importances if available
            feature_importances = None
            if hasattr(sup_model, 'feature_importances_'):
                fi = sup_model.feature_importances_
                feature_importances = [{ 'feature': FEATURE_NAMES[i], 'importance': float(fi[i]) } for i in range(len(FEATURE_NAMES))]

            supervised = {
                'pred_index': sup_pred_idx,
                'label': sup_label,
                'confidence': round(sup_conf, 1),
                'proba': sup_proba,
                'feature_importances': feature_importances
            }
        except Exception:
            supervised = None

    item = {
        'id': uuid.uuid4().hex,
        'filename': fname,
        'cluster_id': int(best),
        'label': label,
        'confidence': round(confidence, 1),
        'features': {k: float(v) for k, v in features.items()},
        'distances': [float(d) for d in dists.tolist()],
        'reasons': reasons,
        'supervised': supervised
    }

    save_history_item(item)

    return render_template('result.html', item=item)


@app.route('/analytics')
def analytics():
    # images expected in static/images/
    imgs = {
        'classifier': url_for('static', filename='images/classifier_results.png'),
        'pca': url_for('static', filename='images/pca_scatter.png'),
        'pie': url_for('static', filename='images/pie_chart.png'),
        'feature_compare': url_for('static', filename='images/feature_compare.png')
    }
    history = read_history()
    return render_template('analytics.html', imgs=imgs, history=history)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='127.0.0.1', port=port, debug=True)
