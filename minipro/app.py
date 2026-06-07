import uuid
from pathlib import Path

from flask import Flask, flash, render_template, request
from werkzeug.utils import secure_filename

from handwriting_supervised import (
    FEATURE_NAMES,
    best_model_name,
    df,
    extract_features,
    predict_new_image_supervised,
    preprocess_image,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_FOLDER = STATIC_DIR / "uploads"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

FEATURE_FEEDBACK = {
    "Spacing (CoV)": {
        "direction": "lower",
        "issue": "Your letter spacing is uneven",
        "tip": "try keeping a steady gap between letters and words by using ruled paper or light pencil guide marks.",
    },
    "Baseline Variation": {
        "direction": "lower",
        "issue": "Your baseline variation is high",
        "tip": "try writing on ruled paper to improve alignment.",
    },
    "Fluency (px/stroke)": {
        "direction": "higher",
        "issue": "Your stroke fluency is low",
        "tip": "try writing slightly slower with continuous strokes instead of lifting the pen often.",
    },
    "Stroke Breaks": {
        "direction": "lower",
        "issue": "Your writing has many stroke breaks",
        "tip": "try maintaining steady pen pressure and completing each letter in fewer broken strokes.",
    },
    "Smoothness (roughness)": {
        "direction": "lower",
        "issue": "Your strokes look rough",
        "tip": "try loosening your grip and practicing smooth curved lines before writing full words.",
    },
}

EXCELLENT_FEATURE_TARGETS = (
    df[df["quality_label"] == "Excellent"][FEATURE_NAMES].mean().to_dict()
)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    app.secret_key = "handwriting-quality-app"

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    @app.route("/", methods=["GET", "POST"])
    def index():
        result = None
        image_url = None
        chart_url = None

        if request.method == "POST":
            uploaded_file = request.files.get("image")

            if not uploaded_file or uploaded_file.filename == "":
                flash("Please choose an image file to upload.")
                return render_template("index.html")

            extension = Path(uploaded_file.filename).suffix.lower()
            if extension not in ALLOWED_EXTENSIONS:
                flash("Please upload a PNG, JPG, JPEG, BMP, or WEBP image.")
                return render_template("index.html")

            safe_name = secure_filename(uploaded_file.filename) or "upload"
            filename = f"{uuid.uuid4().hex}_{Path(safe_name).stem}{extension}"
            saved_path = UPLOAD_FOLDER / filename
            uploaded_file.save(saved_path)
            image_url = f"/static/uploads/{filename}"

            try:
                prediction = build_prediction_result(str(saved_path))
                result = prediction
                if prediction["visualization_path"]:
                    chart_name = Path(prediction["visualization_path"]).name
                    chart_url = f"/static/uploads/{chart_name}"
            except Exception as exc:
                flash(f"Prediction failed: {exc}")
                image_url = None

        return render_template(
            "index.html",
            result=result,
            image_url=image_url,
            chart_url=chart_url,
        )

    return app


def build_prediction_result(image_path: str) -> dict:
    predicted_label, probabilities = predict_new_image_supervised(image_path)
    binary = preprocess_image(image_path)
    features = extract_features(binary)

    visualization_name = f"prediction_{Path(image_path).name}.png"
    visualization_path = UPLOAD_FOLDER / visualization_name
    workspace_chart = BASE_DIR / visualization_name
    if workspace_chart.exists():
        workspace_chart.replace(visualization_path)

    probability_map = {
        label: float(probability)
        for label, probability in zip(["Bad", "Excellent", "Good"], probabilities)
    }
    feature_map = {
        feature_name: float(value)
        for feature_name, value in zip(FEATURE_NAMES, features)
    }
    feedback = build_feature_feedback(predicted_label, feature_map)

    return {
        "label": predicted_label,
        "confidence": max(probabilities) * 100,
        "model_name": best_model_name,
        "probabilities": probability_map,
        "features": feature_map,
        "feedback": feedback,
        "visualization_path": str(visualization_path) if visualization_path.exists() else None,
    }


def build_feature_feedback(predicted_label: str, feature_map: dict) -> list[dict]:
    if predicted_label not in {"Bad", "Good"}:
        return []

    feedback_items = []
    feature_gaps = []

    for feature_name in FEATURE_NAMES:
        value = feature_map[feature_name]
        target = float(EXCELLENT_FEATURE_TARGETS[feature_name])
        rule = FEATURE_FEEDBACK[feature_name]

        if rule["direction"] == "lower":
            gap = (value - target) / target if target else 0
        else:
            gap = (target - value) / target if target else 0

        feature_gaps.append((gap, feature_name, value, target, rule))

    if predicted_label == "Good":
        feature_gaps = sorted(feature_gaps, reverse=True)[:3]

    for gap, feature_name, value, target, rule in feature_gaps:
        needs_work = gap > 0

        if needs_work:
            feedback_items.append(
                {
                    "feature": feature_name,
                    "value": value,
                    "target": target,
                    "message": f"{rule['issue']} — {rule['tip']}",
                }
            )

    return feedback_items


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
