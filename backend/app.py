from flask import Flask, request, jsonify, make_response
from classifier import classify_image
import auto_trainer
import traceback

app = Flask(__name__)

def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

app.after_request(add_cors)


@app.route("/classify", methods=["POST", "OPTIONS"])
def classify():
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file"}), 400
        img_bytes = request.files["image"].read()
        result = classify_image(img_bytes)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/retrain", methods=["POST", "OPTIONS"])
def retrain():
    """
    Called by the frontend immediately after saving a correction.
    Triggers background retraining without blocking the response.
    """
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        status = auto_trainer.get_status()
        if status["is_training"]:
            return jsonify({
                "status": "already_training",
                "message": "Training already in progress, correction queued.",
                "corrections": status["corrections"],
                "classes": status["classes_found"],
            })

        if not status["ready_to_train"]:
            needed_classes = max(0, 2 - len(status["classes_found"]))
            needed_samples = max(0, 4 - status["corrections"])
            msg = []
            if needed_classes > 0:
                msg.append(f"Need {needed_classes} more flower class(es)")
            if needed_samples > 0:
                msg.append(f"Need {needed_samples} more sample(s)")
            return jsonify({
                "status": "not_enough_data",
                "message": " · ".join(msg),
                "corrections": status["corrections"],
                "classes": status["classes_found"],
                "sample_counts": status["sample_counts"],
            })

        # Trigger training in background thread
        auto_trainer.trigger_train()
        return jsonify({
            "status": "training_started",
            "message": "Retraining started in background!",
            "corrections": status["corrections"],
            "classes": status["classes_found"],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/training-status")
def training_status():
    try:
        return jsonify(auto_trainer.get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/flower-categories")
def flower_categories():
    """
    Returns all known flower categories — both base Oxford-102 classes
    and any custom labels added via training corrections.
    Frontend expects: { categories, custom_labels, custom_count }
    """
    try:
        status = auto_trainer.get_status()

        # Custom labels collected from user corrections / added flowers
        custom_labels = sorted(status.get("classes_found", []))
        custom_count  = len(custom_labels)

        # Base classifier categories (Oxford 102 built-ins + custom merged)
        try:
            base_categories = sorted(classify_image.__self__.get_categories()
                                     if hasattr(classify_image, "__self__") else [])
        except Exception:
            base_categories = []

        # Merge: base + custom, deduplicated, sorted
        all_categories = sorted(set(base_categories) | set(custom_labels))

        # Fallback: if classifier exposes no category list, return custom only
        if not all_categories:
            all_categories = custom_labels

        return jsonify({
            "categories":    all_categories,
            "custom_labels": custom_labels,
            "custom_count":  custom_count,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    auto_trainer.start()
    print("Flower API        → http://localhost:5000")
    print("Auto-trainer      → watching Firebase every 30s")
    print("Instant retrain   → POST /retrain (called by frontend)")
    print("Training status   → http://localhost:5000/training-status")
    print("Categories        → http://localhost:5000/flower-categories")
    app.run(host="0.0.0.0", port=5000, debug=False)