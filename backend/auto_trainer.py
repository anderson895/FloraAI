"""
auto_trainer.py
===============
Background thread na nag-mo-monitor ng Firebase flower_training_data.
Auto-retrains ang flower_model.pkl kapag may bagong correction na na-detect.

Ginagamit sa loob ng app.py — hindi kailangang i-run separately.
"""

import os
import sys
import time
import pickle
import threading
import requests
import numpy as np
import cv2
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flower_model.pkl")
FIREBASE_PROJECT = "plantclassification-502b7"
FIRESTORE_BASE   = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT}/databases/(default)/documents"

CHECK_INTERVAL   = 30          # seconds between Firebase polls
MIN_CLASSES      = 2           # minimum unique classes needed to train
MIN_SAMPLES      = 4           # minimum total samples needed to train

# ── State ─────────────────────────────────────────────────────────────────────
_last_known_count  = -1        # how many corrections we saw last time
_training_lock     = threading.Lock()
_is_training       = False


# ── Firebase helpers ──────────────────────────────────────────────────────────
def _fetch_corrections() -> list[tuple[str, str]]:
    """Fetch all docs from flower_training_data. Returns [(label, imageUrl)]."""
    url  = f"{FIRESTORE_BASE}/flower_training_data?pageSize=500"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        docs = r.json().get("documents", [])
        samples = []
        for doc in docs:
            fields = doc.get("fields", {})
            label  = fields.get("label",    {}).get("stringValue", "").strip().lower()
            img_url= fields.get("imageUrl", {}).get("stringValue", "").strip()
            if label and img_url:
                samples.append((label, img_url))
        return samples
    except Exception as e:
        print(f"[auto_trainer] ⚠️  Firebase fetch error: {e}")
        return []


def _fetch_high_conf_classifications() -> list[tuple[str, str]]:
    """Fetch flower_classifications with confidence ≥ 75% as extra training data."""
    url = f"{FIRESTORE_BASE}/flower_classifications?pageSize=500"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        docs = r.json().get("documents", [])
        samples = []
        for doc in docs:
            fields     = doc.get("fields", {})
            label      = fields.get("topFlower", {}).get("stringValue", "").strip().lower()
            img_url    = fields.get("imageUrl",  {}).get("stringValue", "").strip()
            conf_field = fields.get("confidence", {})
            confidence = float(
                conf_field.get("doubleValue",
                conf_field.get("integerValue", 0))
            )
            if label and img_url and confidence >= 75.0:
                samples.append((label, img_url))
        return samples
    except Exception as e:
        print(f"[auto_trainer] ⚠️  Classifications fetch error: {e}")
        return []


def _download_image(url: str) -> np.ndarray | None:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        arr = np.frombuffer(r.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ── Feature extraction (must match classifier.py exactly) ────────────────────
def _extract_features(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (64, 64))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    sky   = (h_ch>=88)&(h_ch<=132)&(s_ch>=15)&(v_ch>=100)
    green = (h_ch>=35)&(h_ch<=88)&(s_ch>=40)
    dark  = v_ch < 30
    fg    = (~(sky|green|dark)).astype(np.uint8) * 255

    h_hist = cv2.calcHist([hsv],[0],fg,[36],[0,180]).flatten()
    s_hist = cv2.calcHist([hsv],[1],fg,[32],[0,256]).flatten()
    v_hist = cv2.calcHist([hsv],[2],fg,[32],[0,256]).flatten()
    h_hist /= (h_hist.sum()+1e-7)
    s_hist /= (s_hist.sum()+1e-7)
    v_hist /= (v_hist.sum()+1e-7)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog  = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
    hf   = hog.compute(gray).flatten()
    hf  /= (np.linalg.norm(hf)+1e-7)

    def mom(arr):
        if len(arr) < 2: return [0.,0.,0.]
        m=arr.mean(); s=arr.std()
        return [m, s, float(np.clip(np.mean(((arr-m)/(s+1e-7))**3),-5,5))]

    fp = fg.flatten() > 0
    hm = mom(h_ch.flatten()[fp]/179.) if fp.any() else [0.,0.,0.]
    sm = mom(s_ch.flatten()[fp]/255.) if fp.any() else [0.,0.,0.]
    vm = mom(v_ch.flatten()[fp]/255.) if fp.any() else [0.,0.,0.]

    cx, cy = 32, 32
    cm = np.zeros((64,64), np.uint8); cv2.circle(cm,(cx,cy),12,255,-1)
    rm = np.zeros((64,64), np.uint8)
    cv2.circle(rm,(cx,cy),28,255,-1); cv2.circle(rm,(cx,cy),12,0,-1)

    def reg(mask):
        px = mask.flatten()>0
        if px.sum()==0: return [0.,0.,0.,0.]
        return [h_ch.flatten()[px].mean()/179., s_ch.flatten()[px].mean()/255.,
                v_ch.flatten()[px].mean()/255., (v_ch.flatten()[px]<80).mean()]

    cf  = reg(cm); rf = reg(rm)
    rad = np.array(cf + rf + [abs(cf[0]-rf[0]), abs(cf[2]-rf[2]), cf[3]])
    return np.concatenate([h_hist, s_hist, v_hist, hf[:50], hm, sm, vm, rad])


def _augment(img: np.ndarray):
    """Yield 8 augmented versions per image."""
    yield img
    yield cv2.flip(img, 1)
    for angle in [-20, 20]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
        yield cv2.warpAffine(img, M, (w, h))
    for gamma in [0.7, 1.4]:
        lut = np.array([((i/255.)**gamma)*255 for i in range(256)], np.uint8)
        yield lut[img]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*1.3, 0, 255)
    yield cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    h, w = img.shape[:2]
    crop = img[h//8:7*h//8, w//8:7*w//8]
    yield cv2.resize(crop, (w, h))


# ── Training ──────────────────────────────────────────────────────────────────
def _do_train(corrections: list, auto_samples: list):
    """Run full retrain using corrections (10x weight) + auto samples (1x)."""
    global _is_training

    try:
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("[auto_trainer] ❌ scikit-learn not installed. Run: pip install scikit-learn")
        return

    print(f"[auto_trainer] 🏋️  Training started — {len(corrections)} corrections + {len(auto_samples)} auto samples")

    X, y = [], []
    failed = 0

    def process(batch, weight, tag):
        nonlocal failed
        for i, (label, url) in enumerate(batch):
            img = _download_image(url)
            if img is None:
                failed += 1
                continue
            for aug in _augment(img):
                try:
                    feat = _extract_features(aug)
                    for _ in range(weight):
                        X.append(feat)
                        y.append(label)
                except Exception:
                    pass

    process(corrections, weight=10, tag="corrections")
    process(auto_samples, weight=1,  tag="auto")

    if len(X) == 0:
        print("[auto_trainer] ❌ No images downloaded — check internet connection")
        return

    X_arr = np.array(X)
    y_arr = np.array(y)

    # Need at least 2 classes
    unique_classes = list(set(y_arr))
    if len(unique_classes) < MIN_CLASSES:
        print(f"[auto_trainer] ⏳ Only {len(unique_classes)} class(es) so far — need {MIN_CLASSES}+ to train")
        print(f"[auto_trainer]    Current classes: {unique_classes}")
        print(f"[auto_trainer]    → Classify more flowers and add corrections for other species!")
        return

    le    = LabelEncoder()
    y_enc = le.fit_transform(y_arr)

    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
              class_weight="balanced", random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  random_state=42, n_jobs=-1)
    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("rf", rf)],
        voting="soft", weights=[2, 1]
    )

    ensemble.fit(X_arr, y_enc)

    model_data = {
        "model":         ensemble,
        "label_encoder": le,
        "classes":       unique_classes,
        "n_classes":     len(unique_classes),
        "n_corrections": len(corrections),
        "n_auto":        len(auto_samples),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f, protocol=4)

    acc = (ensemble.predict(X_arr) == y_enc).mean() * 100
    print(f"[auto_trainer] ✅ Model saved! Classes: {unique_classes} | Train acc: {acc:.1f}%")
    print(f"[auto_trainer] 🔄 classifier.py will auto-reload on next request")


# ── Watcher loop ──────────────────────────────────────────────────────────────
def _watch_loop():
    global _last_known_count, _is_training

    print(f"[auto_trainer] 👀 Watching Firebase every {CHECK_INTERVAL}s for new corrections...")

    while True:
        try:
            corrections = _fetch_corrections()
            current_count = len(corrections)

            if current_count != _last_known_count:
                if _last_known_count == -1:
                    print(f"[auto_trainer] 📊 Found {current_count} existing correction(s) in Firebase")
                else:
                    new = current_count - _last_known_count
                    print(f"[auto_trainer] 🆕 {new} new correction(s) detected! Total: {current_count}")

                _last_known_count = current_count

                # Check if we have enough to train
                counts  = Counter(label for label, _ in corrections)
                classes = list(counts.keys())
                total   = sum(counts.values())

                print(f"[auto_trainer] 📋 Classes: {classes} | Samples: {dict(counts)}")

                if len(classes) < MIN_CLASSES:
                    needed = MIN_CLASSES - len(classes)
                    print(f"[auto_trainer] ⏳ Need {needed} more class(es) before training.")
                    print(f"[auto_trainer]    → Upload & correct {needed} different flower type(s)!")
                elif total < MIN_SAMPLES:
                    needed = MIN_SAMPLES - total
                    print(f"[auto_trainer] ⏳ Need {needed} more sample(s) before training.")
                else:
                    # Enough data — train!
                    if not _training_lock.locked():
                        with _training_lock:
                            _is_training = True
                            auto_samples = _fetch_high_conf_classifications()
                            _do_train(corrections, auto_samples)
                            _is_training = False

        except Exception as e:
            print(f"[auto_trainer] ⚠️  Watcher error: {e}")

        time.sleep(CHECK_INTERVAL)


# ── Public API ────────────────────────────────────────────────────────────────
def start():
    """Start the background watcher thread. Call this from app.py."""
    t = threading.Thread(target=_watch_loop, daemon=True, name="auto_trainer")
    t.start()
    print("[auto_trainer] 🚀 Background trainer started")
    return t


def trigger_train():
    """
    Immediately trigger a retrain in a background thread.
    Called by /retrain endpoint right after frontend saves a correction.
    """
    global _is_training
    if _training_lock.locked():
        print("[auto_trainer] ⏳ Train already running — skipping trigger")
        return

    def _run():
        global _is_training
        with _training_lock:
            _is_training = True
            corrections  = _fetch_corrections()
            auto_samples = _fetch_high_conf_classifications()
            _do_train(corrections, auto_samples)
            _is_training = False

    t = threading.Thread(target=_run, daemon=True, name="auto_trainer_triggered")
    t.start()
    print("[auto_trainer] ⚡ Triggered immediate retrain!")


def is_training() -> bool:
    return _is_training


def get_status() -> dict:
    corrections  = _fetch_corrections()
    counts       = Counter(label for label, _ in corrections)
    model_exists = os.path.exists(MODEL_PATH)
    classes      = []
    if model_exists:
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            classes = data.get("classes", [])
        except Exception:
            pass
    return {
        "corrections":    len(corrections),
        "classes_found":  list(counts.keys()),
        "sample_counts":  dict(counts),
        "model_exists":   model_exists,
        "model_classes":  classes,
        "is_training":    _is_training,
        "ready_to_train": len(counts) >= MIN_CLASSES and sum(counts.values()) >= MIN_SAMPLES,
    }