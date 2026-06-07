import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


CSV_PATH = "handwriting_clusters.csv"
CSV_FALLBACK_PATH = "handwriting_results.csv"
RESULT_IMAGE_PATH = "classifier_results.png"
FEATURE_COLUMNS = [
    "Spacing (CoV)",
    "Baseline Variation",
    "Fluency (px/stroke)",
    "Stroke Breaks",
    "Smoothness (roughness)",
]
CLUSTER_TO_LABEL = {0: "Good", 1: "Excellent", 2: "Bad"}
LABEL_ORDER = ["Bad", "Good", "Excellent"]


@dataclass
class TrainingArtifacts:
    best_model_name: str
    best_model: object
    scaler: StandardScaler
    label_encoder: LabelEncoder
    metrics_df: pd.DataFrame
    cv_scores: Dict[str, float]
    feature_importance: pd.Series


ARTIFACTS: TrainingArtifacts | None = None


def load_and_prepare_data(csv_path: str = CSV_PATH) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Load CSV, map clusters to labels, and encode labels."""
    if not os.path.exists(csv_path):
        if csv_path == CSV_PATH and os.path.exists(CSV_FALLBACK_PATH):
            print(f"Using fallback dataset: {CSV_FALLBACK_PATH}")
            csv_path = CSV_FALLBACK_PATH
        else:
            raise FileNotFoundError(
                f"Could not find '{csv_path}'. Please place handwriting_clusters.csv in the project folder."
            )

    df = pd.read_csv(csv_path)
    missing_columns = [col for col in FEATURE_COLUMNS + ["cluster"] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")

    df = df.copy()
    df["quality_label"] = df["cluster"].map(CLUSTER_TO_LABEL)
    if df["quality_label"].isna().any():
        bad_values = sorted(df.loc[df["quality_label"].isna(), "cluster"].unique().tolist())
        raise ValueError(f"Found unmapped cluster values: {bad_values}. Expected only 0, 1, 2.")

    label_encoder = LabelEncoder()
    label_encoder.fit(LABEL_ORDER)
    y = label_encoder.transform(df["quality_label"])
    return df, y, label_encoder


def split_and_scale_data(
    df: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split data with stratification and fit StandardScaler only on train data."""
    X = df[FEATURE_COLUMNS].values
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_models(random_state: int = 42) -> Dict[str, object]:
    """Define supervised models to train and compare."""
    return {
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        ),
    }


def evaluate_models(
    models: Dict[str, object],
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, np.ndarray], Dict[str, float], str]:
    """Train models and compute test metrics, confusion matrices, and CV F1 scores."""
    trained_models = {}
    confusion_matrices = {}
    cv_scores = {}
    metric_rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        metric_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
        )

        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)
        cv_scores[model_name] = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=skf,
            scoring="f1_weighted",
        ).mean()
        trained_models[model_name] = model

    metrics_df = pd.DataFrame(metric_rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    best_model_name = metrics_df.iloc[0]["Model"]
    return trained_models, metrics_df, confusion_matrices, cv_scores, best_model_name


def create_results_visualization(
    metrics_df: pd.DataFrame,
    confusion_matrices: Dict[str, np.ndarray],
    cv_scores: Dict[str, float],
    feature_importance: pd.Series,
    label_encoder: LabelEncoder,
    best_model_name: str,
    output_path: str = RESULT_IMAGE_PATH,
) -> None:
    """Create and save a multi-panel summary figure."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    class_names = label_encoder.classes_

    for ax, model_name in zip(axes.flatten()[:3], metrics_df["Model"].tolist()):
        sns.heatmap(
            confusion_matrices[model_name],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    metrics_plot_df = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(data=metrics_plot_df, x="Model", y="Score", hue="Metric", ax=axes[1, 1])
    axes[1, 1].set_title("Metric Comparison")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].tick_params(axis="x", rotation=15)

    cv_df = pd.DataFrame({"Model": list(cv_scores.keys()), "CV F1 Score": list(cv_scores.values())})
    sns.barplot(data=cv_df, x="Model", y="CV F1 Score", ax=axes[2, 0], palette="viridis")
    axes[2, 0].set_title("5-Fold Cross-Validation F1 Comparison")
    axes[2, 0].set_ylim(0, 1.05)
    axes[2, 0].tick_params(axis="x", rotation=15)

    feature_importance.sort_values(ascending=True).plot(kind="barh", ax=axes[2, 1], color="#4C72B0")
    axes[2, 1].set_title("Random Forest Feature Importance")
    axes[2, 1].set_xlabel("Importance")

    # Add a compact summary table with best-model highlighting.
    summary_ax = fig.add_axes([0.12, 0.01, 0.8, 0.14])
    summary_ax.axis("off")
    summary_df = metrics_df.copy()
    summary_df["CV F1 Score"] = summary_df["Model"].map(cv_scores)
    summary_df = summary_df.round(4)
    table = summary_ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    best_row_idx = summary_df.index[summary_df["Model"] == best_model_name][0] + 1
    for col_idx in range(len(summary_df.columns)):
        table[(best_row_idx, col_idx)].set_facecolor("#C8E6C9")

    plt.tight_layout(rect=[0, 0.16, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def preprocess_handwriting_image(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read and binarize handwriting image for feature extraction."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return image, gray, binary


def _spacing_cov(binary: np.ndarray) -> float:
    projection = np.sum(binary > 0, axis=0)
    ink_columns = projection > 0
    gaps = []
    gap_length = 0

    for has_ink in ink_columns:
        if not has_ink:
            gap_length += 1
        elif gap_length > 0:
            gaps.append(gap_length)
            gap_length = 0

    if gap_length > 0:
        gaps.append(gap_length)

    if len(gaps) < 2:
        return 0.0
    return float(np.std(gaps) / (np.mean(gaps) + 1e-6))


def _baseline_variation(binary: np.ndarray) -> float:
    baseline_points = []
    for col_idx in range(binary.shape[1]):
        ink_rows = np.where(binary[:, col_idx] > 0)[0]
        if ink_rows.size > 0:
            baseline_points.append(ink_rows.max())
    if len(baseline_points) < 2:
        return 0.0
    return float(np.std(baseline_points))


def _fluency(binary: np.ndarray) -> float:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    stroke_count = max(len(contours), 1)
    contour_length = sum(cv2.arcLength(cnt, False) for cnt in contours)
    return float(contour_length / stroke_count)


def _stroke_breaks(binary: np.ndarray) -> float:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0.0

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(5, int(0.0002 * binary.size))
    return float(np.sum(component_areas >= min_area))


def _smoothness(gray: np.ndarray, binary: np.ndarray) -> float:
    if np.count_nonzero(binary) == 0:
        return 0.0
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    ink_values = np.abs(laplacian[binary > 0])
    return float(np.mean(ink_values))


def extract_handwriting_features(image_path: str, show_visualization: bool = True) -> pd.DataFrame:
    """Extract the five handwriting quality features from a single image."""
    original, gray, binary = preprocess_handwriting_image(image_path)

    features = {
        "Spacing (CoV)": _spacing_cov(binary),
        "Baseline Variation": _baseline_variation(binary),
        "Fluency (px/stroke)": _fluency(binary),
        "Stroke Breaks": _stroke_breaks(binary),
        "Smoothness (roughness)": _smoothness(gray, binary),
    }

    if show_visualization:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[1].imshow(gray, cmap="gray")
        axes[1].set_title("Grayscale")
        axes[2].imshow(binary, cmap="gray")
        axes[2].set_title("Binary Ink Mask")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


def train_handwriting_quality_classifier(csv_path: str = CSV_PATH) -> TrainingArtifacts:
    """Run the complete training/evaluation pipeline and store the best model."""
    global ARTIFACTS

    df, y, label_encoder = load_and_prepare_data(csv_path)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(df, y)

    models = build_models()
    trained_models, metrics_df, confusion_matrices, cv_scores, best_model_name = evaluate_models(
        models,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
    )

    rf_model = trained_models["Random Forest"]
    feature_importance = pd.Series(rf_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(
        ascending=False
    )

    create_results_visualization(
        metrics_df=metrics_df,
        confusion_matrices=confusion_matrices,
        cv_scores=cv_scores,
        feature_importance=feature_importance,
        label_encoder=label_encoder,
        best_model_name=best_model_name,
        output_path=RESULT_IMAGE_PATH,
    )

    ARTIFACTS = TrainingArtifacts(
        best_model_name=best_model_name,
        best_model=trained_models[best_model_name],
        scaler=scaler,
        label_encoder=label_encoder,
        metrics_df=metrics_df,
        cv_scores=cv_scores,
        feature_importance=feature_importance,
    )

    best_metrics = metrics_df.loc[metrics_df["Model"] == best_model_name].iloc[0]
    print("\nBest Model Selected")
    print(f"Model: {best_model_name}")
    print(f"Accuracy:  {best_metrics['Accuracy']:.4f}")
    print(f"Precision: {best_metrics['Precision']:.4f}")
    print(f"Recall:    {best_metrics['Recall']:.4f}")
    print(f"F1 Score:  {best_metrics['F1 Score']:.4f}")
    print(f"CV F1:     {cv_scores[best_model_name]:.4f}")
    print(f"Saved visualization to: {RESULT_IMAGE_PATH}")

    return ARTIFACTS


def predict_new_image_supervised(image_path: str) -> str:
    """
    Predict handwriting quality for a new image using the trained best model.

    Example:
        predict_new_image_supervised("handwriting_400/1.jpg")
    """
    global ARTIFACTS

    if ARTIFACTS is None:
        ARTIFACTS = train_handwriting_quality_classifier(CSV_PATH)

    try:
        feature_df = extract_handwriting_features(image_path, show_visualization=False)
        scaled_features = ARTIFACTS.scaler.transform(feature_df.values)
        predicted_label_idx = int(ARTIFACTS.best_model.predict(scaled_features)[0])
        predicted_label = ARTIFACTS.label_encoder.inverse_transform([predicted_label_idx])[0]

        if hasattr(ARTIFACTS.best_model, "predict_proba"):
            probabilities = ARTIFACTS.best_model.predict_proba(scaled_features)[0]
        else:
            decision = ARTIFACTS.best_model.decision_function(scaled_features)[0]
            exp_scores = np.exp(decision - np.max(decision))
            probabilities = exp_scores / np.sum(exp_scores)

        probability_df = pd.DataFrame(
            {
                "Quality": ARTIFACTS.label_encoder.classes_,
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False)

        print("\nPrediction Result")
        print(f"Image: {image_path}")
        print(f"Best Model Used: {ARTIFACTS.best_model_name}")
        print(f"Predicted Quality: {predicted_label}")
        print("\nProbability Scores")
        print(probability_df.to_string(index=False, formatters={"Probability": "{:.4f}".format}))
        print("\nExtracted Features")
        print(feature_df.to_string(index=False, formatters={col: "{:.4f}".format for col in FEATURE_COLUMNS}))

        return predicted_label
    except (FileNotFoundError, ValueError) as exc:
        print(f"Prediction error: {exc}")
        return "Prediction Failed"


def main() -> None:
    """Entry point for training and terminal-based single-image prediction."""
    try:
        train_handwriting_quality_classifier(CSV_PATH)

        image_path = input(
            "\nEnter handwriting image path for prediction "
            "(example: handwriting_400/1.jpg), or press Enter to exit: "
        ).strip()

        if image_path:
            predict_new_image_supervised(image_path)
        else:
            print("No image path entered. Exiting after training.")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Pipeline error: {exc}")


if __name__ == "__main__":
    main()
