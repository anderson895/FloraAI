"""
Flower Classifier
=================
Priority order:
  1. Correction cache  — visual similarity match vs Firebase corrections
                         Works WITHOUT a trained model. 3+ corrections = override.
  2. ML model          — SVM+RF ensemble (flower_model.pkl)
  3. Rule-based HSV    — fallback, uses BUILT-IN + FIREBASE CUSTOM profiles

Dynamic custom flowers:
  • Any label saved to Firebase flower_training_data that is NOT in BUILT_IN_PROFILES
    gets auto-built into a runtime profile from its sample images.
  • No code changes needed — just save corrections via the UI.
"""
import cv2
import numpy as np
import os
import pickle
import time
import requests
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flower_model.pkl")
FIREBASE_URL = ("https://firestore.googleapis.com/v1/projects/"
                "plantclassification-502b7/databases/(default)/documents/"
                "flower_training_data?pageSize=500")

# ── Flower metadata (icon + bbox color) ──────────────────────────────────────
FLOWER_META = {
    "pink primrose":             {"icon":"🌸","color":(200,100,200)},
    "hard-leaved pocket orchid": {"icon":"🌸","color":(180,80,200)},
    "canterbury bells":          {"icon":"🔔","color":(100,120,220)},
    "sweet pea":                 {"icon":"🌸","color":(180,130,200)},
    "wild geranium":             {"icon":"🌸","color":(160,80,180)},
    "tiger lily":                {"icon":"🌸","color":(0,140,240)},
    "moon orchid":               {"icon":"🌸","color":(210,210,210)},
    "bird of paradise":          {"icon":"🌺","color":(0,170,255)},
    "monkshood":                 {"icon":"💜","color":(80,60,200)},
    "globe thistle":             {"icon":"💜","color":(100,80,200)},
    "snapdragon":                {"icon":"🌸","color":(180,60,160)},
    "colt's foot":               {"icon":"🌼","color":(0,200,220)},
    "king protea":               {"icon":"🌺","color":(60,60,200)},
    "spear thistle":             {"icon":"💜","color":(120,70,210)},
    "yellow iris":               {"icon":"💛","color":(0,200,255)},
    "globe-flower":              {"icon":"🌼","color":(0,190,255)},
    "purple coneflower":         {"icon":"💜","color":(140,70,200)},
    "peruvian lily":             {"icon":"🌸","color":(0,160,255)},
    "balloon flower":            {"icon":"💜","color":(100,100,210)},
    "giant white arum lily":     {"icon":"🤍","color":(220,220,220)},
    "fire lily":                 {"icon":"🌺","color":(0,80,230)},
    "pincushion flower":         {"icon":"💜","color":(120,90,210)},
    "fritillary":                {"icon":"🌸","color":(130,80,200)},
    "red ginger":                {"icon":"🌺","color":(30,40,220)},
    "grape hyacinth":            {"icon":"💜","color":(80,60,210)},
    "corn poppy":                {"icon":"🌹","color":(40,40,220)},
    "prince of wales feathers":  {"icon":"🌸","color":(200,190,200)},
    "stemless gentian":          {"icon":"💙","color":(60,60,230)},
    "artichoke":                 {"icon":"🌿","color":(80,160,80)},
    "sweet william":             {"icon":"🌸","color":(140,60,200)},
    "carnation":                 {"icon":"🌸","color":(130,60,200)},
    "garden phlox":              {"icon":"🌸","color":(180,90,200)},
    "love in the mist":          {"icon":"💙","color":(100,130,210)},
    "cosmos":                    {"icon":"🌸","color":(180,80,200)},
    "alpine sea holly":          {"icon":"💙","color":(80,120,220)},
    "ruby-lipped cattleya":      {"icon":"🌸","color":(160,60,200)},
    "cape flower":               {"icon":"🌸","color":(100,180,220)},
    "great masterwort":          {"icon":"🌸","color":(200,170,210)},
    "siam tulip":                {"icon":"🌷","color":(180,100,200)},
    "lenten rose":               {"icon":"🌸","color":(140,80,200)},
    "barberton daisy":           {"icon":"🌼","color":(0,160,255)},
    "daffodil":                  {"icon":"🌼","color":(0,210,255)},
    "sword lily":                {"icon":"🌸","color":(0,130,240)},
    "poinsettia":                {"icon":"🌺","color":(30,30,220)},
    "bolero deep blue":          {"icon":"💙","color":(60,40,230)},
    "wallflower":                {"icon":"🌼","color":(0,150,240)},
    "marigold":                  {"icon":"🌼","color":(0,140,255)},
    "buttercup":                 {"icon":"🌼","color":(0,220,255)},
    "daisy":                     {"icon":"🌼","color":(50,200,50)},
    "common dandelion":          {"icon":"🌼","color":(0,210,210)},
    "petunia":                   {"icon":"🌸","color":(160,70,200)},
    "wild pansy":                {"icon":"💜","color":(100,80,210)},
    "primula":                   {"icon":"🌸","color":(180,80,200)},
    "sunflower":                 {"icon":"🌻","color":(0,200,255)},
    "lilac hibiscus":            {"icon":"🌺","color":(150,80,210)},
    "bishop of llandaff":        {"icon":"🌺","color":(30,30,230)},
    "gaura":                     {"icon":"🤍","color":(210,200,220)},
    "geranium":                  {"icon":"🌸","color":(140,60,210)},
    "orange dahlia":             {"icon":"🌸","color":(0,130,255)},
    "pink-yellow dahlia":        {"icon":"🌸","color":(100,180,230)},
    "cautleya spicata":          {"icon":"🌸","color":(0,180,230)},
    "japanese anemone":          {"icon":"🌸","color":(190,100,210)},
    "black-eyed susan":          {"icon":"🌼","color":(0,180,240)},
    "silverbush":                {"icon":"🤍","color":(200,210,210)},
    "californian poppy":         {"icon":"🌼","color":(0,150,255)},
    "osteospermum":              {"icon":"🌸","color":(180,100,210)},
    "spring crocus":             {"icon":"💜","color":(110,80,210)},
    "iris":                      {"icon":"💙","color":(90,80,220)},
    "windflower":                {"icon":"🌸","color":(200,130,210)},
    "tree poppy":                {"icon":"🤍","color":(210,200,200)},
    "gazania":                   {"icon":"🌼","color":(0,160,255)},
    "azalea":                    {"icon":"🌸","color":(170,70,200)},
    "water lily":                {"icon":"🪷","color":(100,200,220)},
    "rose":                      {"icon":"🌹","color":(50,50,220)},
    "thorn apple":               {"icon":"🤍","color":(210,210,200)},
    "morning glory":             {"icon":"💜","color":(90,80,220)},
    "passion flower":            {"icon":"🌸","color":(120,80,210)},
    "lotus":                     {"icon":"🪷","color":(180,100,180)},
    "toad lily":                 {"icon":"🌸","color":(150,90,210)},
    "anthurium":                 {"icon":"🌺","color":(30,40,230)},
    "frangipani":                {"icon":"🌸","color":(200,200,210)},
    "clematis":                  {"icon":"💜","color":(100,70,220)},
    "hibiscus":                  {"icon":"🌺","color":(30,30,230)},
    "columbine":                 {"icon":"💜","color":(110,70,220)},
    "desert-rose":               {"icon":"🌸","color":(180,80,200)},
    "tree mallow":               {"icon":"🌸","color":(160,80,210)},
    "magnolia":                  {"icon":"🤍","color":(220,200,210)},
    "cyclamen":                  {"icon":"🌸","color":(190,80,200)},
    "watercress":                {"icon":"🌿","color":(60,180,80)},
    "canna lily":                {"icon":"🌺","color":(0,100,240)},
    "hippeastrum":               {"icon":"🌺","color":(40,40,230)},
    "bee balm":                  {"icon":"🌺","color":(50,50,230)},
    "pink quill":                {"icon":"🌸","color":(190,100,210)},
    "foxglove":                  {"icon":"🌸","color":(150,80,210)},
    "bougainvillea":             {"icon":"🌺","color":(30,30,220)},
    "camellia":                  {"icon":"🌺","color":(40,40,200)},
    "mallow":                    {"icon":"🌸","color":(170,80,200)},
    "mexican petunia":           {"icon":"💜","color":(100,70,220)},
    "bromelia":                  {"icon":"🌺","color":(40,60,230)},
    "blanket flower":            {"icon":"🌼","color":(0,140,250)},
    "trumpet creeper":           {"icon":"🌺","color":(0,100,240)},
    "blackberry lily":           {"icon":"🌼","color":(0,150,250)},
    "common tulip":              {"icon":"🌷","color":(180,80,210)},
    "wild rose":                 {"icon":"🌹","color":(160,60,200)},
}

def _meta(label: str) -> dict:
    key = next((k for k in FLOWER_META if k.lower() == label.lower()), None)
    return FLOWER_META[key] if key else {"icon":"🌸","color":(100,255,100)}


# ── Built-in HSV profiles ─────────────────────────────────────────────────────
BUILT_IN_PROFILES = {
    "sunflower":         {"dominant_hue":(25,15),"dominant_hue2":None,"saturation_min":120,"has_dark_center":True, "high_texture":False,"low_texture":False},
    "marigold":          {"dominant_hue":(18,12),"dominant_hue2":None,"saturation_min":140,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "buttercup":         {"dominant_hue":(28,10),"dominant_hue2":None,"saturation_min":150,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "daffodil":          {"dominant_hue":(27,10),"dominant_hue2":None,"saturation_min":100,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "common dandelion":  {"dominant_hue":(27,10),"dominant_hue2":None,"saturation_min":100,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "yellow iris":       {"dominant_hue":(27,10),"dominant_hue2":None,"saturation_min":120,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "californian poppy": {"dominant_hue":(18,10),"dominant_hue2":None,"saturation_min":150,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "corn poppy":        {"dominant_hue":(0, 8), "dominant_hue2":None,"saturation_min":160,"has_dark_center":True, "high_texture":False,"low_texture":False},
    "rose":              {"dominant_hue":(160,20),"dominant_hue2":(0,10),"saturation_min":80,"has_dark_center":False,"high_texture":True, "low_texture":False},
    "poinsettia":        {"dominant_hue":(0, 8), "dominant_hue2":None,"saturation_min":140,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "anthurium":         {"dominant_hue":(0, 8), "dominant_hue2":None,"saturation_min":160,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "fire lily":         {"dominant_hue":(5, 10),"dominant_hue2":None,"saturation_min":160,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "pink primrose":     {"dominant_hue":(165,10),"dominant_hue2":None,"saturation_min":80, "has_dark_center":False,"high_texture":False,"low_texture":False},
    "cosmos":            {"dominant_hue":(163,12),"dominant_hue2":None,"saturation_min":80, "has_dark_center":False,"high_texture":False,"low_texture":False},
    "petunia":           {"dominant_hue":(155,15),"dominant_hue2":None,"saturation_min":80, "has_dark_center":False,"high_texture":False,"low_texture":False},
    "camellia":          {"dominant_hue":(0, 10),"dominant_hue2":None,"saturation_min":80, "has_dark_center":False,"high_texture":False,"low_texture":False},
    "bougainvillea":     {"dominant_hue":(158,12),"dominant_hue2":None,"saturation_min":100,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "hibiscus":          {"dominant_hue":(0, 12),"dominant_hue2":None,"saturation_min":120,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "iris":              {"dominant_hue":(120,20),"dominant_hue2":None,"saturation_min":60, "has_dark_center":False,"high_texture":False,"low_texture":False},
    "wild pansy":        {"dominant_hue":(125,20),"dominant_hue2":None,"saturation_min":80, "has_dark_center":False,"high_texture":False,"low_texture":False},
    "grape hyacinth":    {"dominant_hue":(120,15),"dominant_hue2":None,"saturation_min":100,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "purple coneflower": {"dominant_hue":(155,15),"dominant_hue2":None,"saturation_min":60, "has_dark_center":True, "high_texture":False,"low_texture":False},
    "clematis":          {"dominant_hue":(125,20),"dominant_hue2":None,"saturation_min":60, "has_dark_center":False,"high_texture":False,"low_texture":False},
    "daisy":             {"dominant_hue":(0,180), "dominant_hue2":None,"saturation_min":0,  "saturation_max":50,"has_dark_center":True,"high_texture":False,"low_texture":False},
    "magnolia":          {"dominant_hue":(0,180), "dominant_hue2":None,"saturation_min":0,  "saturation_max":60,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "frangipani":        {"dominant_hue":(0,180), "dominant_hue2":None,"saturation_min":0,  "saturation_max":70,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "moon orchid":       {"dominant_hue":(0,180), "dominant_hue2":None,"saturation_min":0,  "saturation_max":40,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "tiger lily":        {"dominant_hue":(12, 8), "dominant_hue2":None,"saturation_min":160,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "orange dahlia":     {"dominant_hue":(10,10), "dominant_hue2":None,"saturation_min":140,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "bird of paradise":  {"dominant_hue":(15, 8), "dominant_hue2":None,"saturation_min":160,"has_dark_center":False,"high_texture":False,"low_texture":False},
    "lotus":             {"dominant_hue":(163,12),"dominant_hue2":None,"saturation_min":40, "has_dark_center":False,"high_texture":False,"low_texture":False},
    "water lily":        {"dominant_hue":(163,12),"dominant_hue2":None,"saturation_min":30, "has_dark_center":False,"high_texture":False,"low_texture":True},
}


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC CUSTOM PROFILE BUILDER
# Automatically builds HSV profiles from Firebase sample images.
# ══════════════════════════════════════════════════════════════════════════════

_custom_profiles: dict     = {}   # { label: profile_dict }
_custom_profiles_time: float = 0.0
_CUSTOM_TTL = 120.0  # rebuild every 2 minutes

def _img_from_url(url: str) -> Optional[np.ndarray]:
    try:
        r = requests.get(url, timeout=10)
        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def _build_profile_from_images(imgs: list) -> dict:
    """
    Auto-build an HSV profile by analysing sample images.
    Computes dominant foreground hue, saturation stats and texture variance.
    """
    h_vals, s_vals, v_vals, tex_vals = [], [], [], []
    for img in imgs:
        img_r = cv2.resize(img, (64, 64))
        hsv   = cv2.cvtColor(img_r, cv2.COLOR_BGR2HSV)
        h_ch  = hsv[:,:,0]; s_ch = hsv[:,:,1]; v_ch = hsv[:,:,2]
        green = (h_ch>=35)&(h_ch<=88)&(s_ch>=40)
        dark  = v_ch < 30
        fg    = ~(green|dark)
        if fg.sum() < 20:
            fg = np.ones_like(h_ch, bool)
        h_vals.append(float(h_ch[fg].mean()))
        s_vals.append(float(s_ch[fg].mean()))
        v_vals.append(float(v_ch[fg].mean()))
        gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        tex_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    mean_h   = float(np.mean(h_vals))
    mean_s   = float(np.mean(s_vals))
    mean_tex = float(np.mean(tex_vals))

    # Hue tolerance based on spread across samples
    hue_spread = max(10, int(np.std(h_vals) * 1.5))
    hue_spread = min(hue_spread, 25)  # cap at ±25

    return {
        "dominant_hue":   (int(mean_h), hue_spread),
        "dominant_hue2":  None,
        "saturation_min": max(0, int(mean_s * 0.6)),
        "saturation_max": 256,
        "has_dark_center":False,
        "high_texture":   mean_tex > 700,
        "low_texture":    mean_tex < 300,
        "_auto":          True,   # marker — auto-built profile
        "_sample_count":  len(imgs),
    }

def _load_custom_profiles():
    """
    Pulls all Firebase corrections, groups by label.
    For labels NOT in BUILT_IN_PROFILES, builds a dynamic HSV profile.
    """
    global _custom_profiles, _custom_profiles_time
    now = time.time()
    if now - _custom_profiles_time < _CUSTOM_TTL and _custom_profiles:
        return

    try:
        r = requests.get(FIREBASE_URL, timeout=10)
        r.raise_for_status()
        docs = r.json().get("documents", [])

        # Group image URLs by label
        label_urls: dict = {}
        for doc in docs:
            f   = doc.get("fields", {})
            lbl = f.get("label", {}).get("stringValue","").strip().lower()
            url = f.get("imageUrl",{}).get("stringValue","").strip()
            if lbl and url:
                label_urls.setdefault(lbl, []).append(url)

        new_custom = {}
        for label, urls in label_urls.items():
            if label in BUILT_IN_PROFILES:
                continue  # already handled by built-in
            # Need at least 2 samples to build a reliable profile
            if len(urls) < 2:
                continue
            imgs = [img for url in urls[:8] if (img := _img_from_url(url)) is not None]
            if len(imgs) < 2:
                continue
            profile = _build_profile_from_images(imgs)
            new_custom[label] = profile
            print(f"[classifier] 🌸 Custom profile built: '{label}' "
                  f"(H={profile['dominant_hue']}, n={profile['_sample_count']})")

        _custom_profiles = new_custom
        _custom_profiles_time = now

    except Exception as e:
        print(f"[classifier] ⚠️  Custom profile load failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CORRECTION CACHE  — highest priority, works without trained model
# ══════════════════════════════════════════════════════════════════════════════

_correction_cache: dict  = {}   # { label: [feature_vec, ...] }
_correction_cache_time: float = 0.0
_CACHE_TTL = 60.0

def _extract_features(img_bgr: np.ndarray) -> np.ndarray:
    img   = cv2.resize(img_bgr, (32, 32))
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh    = cv2.calcHist([hsv],[0],None,[18],[0,180]).flatten()
    sh    = cv2.calcHist([hsv],[1],None,[16],[0,256]).flatten()
    vh    = cv2.calcHist([hsv],[2],None,[16],[0,256]).flatten()
    hh   /= (hh.sum()+1e-7); sh /= (sh.sum()+1e-7); vh /= (vh.sum()+1e-7)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tex   = np.array([cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0])
    return np.concatenate([hh, sh, vh, tex])

def _load_correction_cache():
    global _correction_cache, _correction_cache_time
    now = time.time()
    if now - _correction_cache_time < _CACHE_TTL and _correction_cache:
        return
    try:
        r    = requests.get(FIREBASE_URL, timeout=10)
        docs = r.json().get("documents", [])
        new: dict = {}
        for doc in docs:
            f   = doc.get("fields",{})
            lbl = f.get("label",{}).get("stringValue","").strip().lower()
            url = f.get("imageUrl",{}).get("stringValue","").strip()
            if not lbl or not url:
                continue
            img = _img_from_url(url)
            if img is None:
                continue
            new.setdefault(lbl, []).append(_extract_features(img))
        _correction_cache = new
        _correction_cache_time = now
        print(f"[classifier] 🔄 Cache: { {k:len(v) for k,v in new.items()} }")
    except Exception as e:
        print(f"[classifier] ⚠️  Cache load error: {e}")

def _correction_override(img_bgr: np.ndarray) -> Optional[Tuple[str, float]]:
    if not _correction_cache:
        return None
    feat = _extract_features(img_bgr)
    best_label, best_score = None, -1.0
    for label, feats in _correction_cache.items():
        if not feats:
            continue
        sims = [np.dot(feat, cf) / (np.linalg.norm(feat)*np.linalg.norm(cf)+1e-7)
                for cf in feats]
        avg  = float(np.mean(sims))
        w    = min(len(feats)/3.0, 2.0)
        ws   = avg * w
        if ws > best_score:
            best_score, best_label = ws, label
    n         = len(_correction_cache.get(best_label, []))
    threshold = 0.85 if n >= 5 else 0.90 if n >= 3 else 1.1
    if best_label and best_score >= threshold:
        conf = min(round(best_score*50, 1), 95.0)
        print(f"[classifier] 🎯 Override → '{best_label}' (sim={best_score:.3f}, n={n})")
        return best_label, conf
    return None


# ── ML model cache ─────────────────────────────────────────────────────────────
_model_cache  = None
_model_mtime  = None

def _load_model():
    global _model_cache, _model_mtime
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        mtime = os.path.getmtime(MODEL_PATH)
        if _model_cache is None or mtime != _model_mtime:
            with open(MODEL_PATH,"rb") as f:
                _model_cache = pickle.load(f)
            _model_mtime = mtime
            print(f"[classifier] ✅ ML model — {len(_model_cache['label_encoder'].classes_)} classes")
        return _model_cache
    except Exception as e:
        print(f"[classifier] ❌ Model load error: {e}")
        return None


# ── Feature extraction for ML ──────────────────────────────────────────────────
def extract_features_for_model(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr,(64,64))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch,s_ch,v_ch = hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    sky   = (h_ch>=88)&(h_ch<=132)&(s_ch>=15)&(v_ch>=100)
    green = (h_ch>=35)&(h_ch<=88)&(s_ch>=40)
    dark  = v_ch < 30
    fg    = (~(sky|green|dark)).astype(np.uint8)*255
    h_hist = cv2.calcHist([hsv],[0],fg,[36],[0,180]).flatten()
    s_hist = cv2.calcHist([hsv],[1],fg,[32],[0,256]).flatten()
    v_hist = cv2.calcHist([hsv],[2],fg,[32],[0,256]).flatten()
    h_hist/=(h_hist.sum()+1e-7); s_hist/=(s_hist.sum()+1e-7); v_hist/=(v_hist.sum()+1e-7)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog  = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
    hf   = hog.compute(gray).flatten(); hf/=(np.linalg.norm(hf)+1e-7)
    def mom(arr):
        if len(arr)<2: return [0.,0.,0.]
        m=arr.mean(); s=arr.std()
        return [m,s,float(np.clip(np.mean(((arr-m)/(s+1e-7))**3),-5,5))]
    fp=fg.flatten()>0
    hm=mom(h_ch.flatten()[fp]/179.) if fp.any() else [0.,0.,0.]
    sm=mom(s_ch.flatten()[fp]/255.) if fp.any() else [0.,0.,0.]
    vm=mom(v_ch.flatten()[fp]/255.) if fp.any() else [0.,0.,0.]
    cx,cy=32,32
    cm=np.zeros((64,64),np.uint8); cv2.circle(cm,(cx,cy),12,255,-1)
    rm=np.zeros((64,64),np.uint8); cv2.circle(rm,(cx,cy),28,255,-1); cv2.circle(rm,(cx,cy),12,0,-1)
    def reg(mask):
        px=mask.flatten()>0
        if px.sum()==0: return [0.,0.,0.,0.]
        return [h_ch.flatten()[px].mean()/179.,s_ch.flatten()[px].mean()/255.,
                v_ch.flatten()[px].mean()/255.,(v_ch.flatten()[px]<80).mean()]
    cf=reg(cm); rf=reg(rm)
    rad=np.array(cf+rf+[abs(cf[0]-rf[0]),abs(cf[2]-rf[2]),cf[3]])
    return np.concatenate([h_hist,s_hist,v_hist,hf[:50],hm,sm,vm,rad])

def predict_with_model(img_bgr: np.ndarray) -> List[Tuple[str,float]]:
    data = _load_model()
    if data is None: return []
    feat  = extract_features_for_model(img_bgr).reshape(1,-1)
    probs = data["model"].predict_proba(feat)[0]
    pairs = sorted(zip(data["label_encoder"].classes_, probs.tolist()), key=lambda x:-x[1])
    return [(str(l),float(p)) for l,p in pairs]


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class BoundingBox:
    x:int; y:int; w:int; h:int

@dataclass
class Detection:
    flower:str; confidence:float; bbox:BoundingBox
    icon:str; color:Tuple[int,int,int]
    all_predictions:List[Tuple[str,float]]


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED SCORING  (built-in + dynamic custom profiles)
# ══════════════════════════════════════════════════════════════════════════════

def _hue_frac(hsv:np.ndarray, hc:int, ht:int, smin:int=40, vmin:int=40) -> float:
    h=hsv[:,:,0]; s=hsv[:,:,1]; v=hsv[:,:,2]
    if hc <= ht:
        hm = (h <= hc+ht) | (h >= 180-ht+hc)
    elif hc >= 180-ht:
        hm = (h >= hc-ht) | (h <= ht-(180-hc))
    else:
        hm = (h >= hc-ht) & (h <= hc+ht)
    return float(np.count_nonzero(hm&(s>=smin)&(v>=vmin))) / (h.size+1e-6)

def _rulebased_scores(img_bgr:np.ndarray, bbox:BoundingBox) -> List[Tuple[str,float]]:
    x,y,w,h = bbox.x,bbox.y,bbox.w,bbox.h
    roi = img_bgr[y:y+h, x:x+w]
    if roi.size==0: return []
    r   = cv2.resize(roi,(128,128))
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    H,W = 128,128
    hc=hsv[:,:,0]; sc=hsv[:,:,1]; vc=hsv[:,:,2]
    fg = ~( ((hc>=88)&(hc<=132)&(sc>=15)&(vc>=100)) | ((hc>=35)&(hc<=88)&(sc>=40)) | (vc<25) )
    fp = fg.sum()
    if fp < 50: fg = np.ones((H,W),bool)
    ms = float(sc[fg].mean()) if fp>0 else 0.0
    mv = float(vc[fg].mean()) if fp>0 else 0.0
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    tex  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Merge built-in and custom profiles
    all_profiles = {**BUILT_IN_PROFILES, **_custom_profiles}

    scores = {}
    for flower, p in all_profiles.items():
        sc_val = 0.0; pen = 0.0
        hcenter,htol = p["dominant_hue"]
        smin = p.get("saturation_min",60)
        smax = p.get("saturation_max",256)

        # 1. Petal color (50 pts)
        frac = _hue_frac(hsv*fg[:,:,np.newaxis], hcenter, htol, max(smin-20,0), 40)
        if p.get("dominant_hue2"):
            hc2,ht2 = p["dominant_hue2"]
            frac = max(frac, _hue_frac(hsv*fg[:,:,np.newaxis], hc2, ht2, max(smin-20,0), 40))
        sc_val += frac*50

        # 2. Saturation (15 pts)
        if smin <= ms <= smax: sc_val += 15
        elif ms < smin-40:     pen    += 20

        # 3. Texture (20 pts)
        if p.get("high_texture"):
            sc_val += 20 if tex>800 else (10 if tex>400 else 0); pen += (25 if tex<300 else 0)
        if p.get("low_texture"):
            sc_val += 15 if tex<400 else 0; pen += (20 if tex>800 else 0)

        # 4. White/pale special case
        if smax < 80:
            pale = float(np.count_nonzero(fg&(sc<60)&(vc>160)))/(fp+1)
            sc_val += pale*40
            if ms > 100: pen += 30

        # 5. Brightness (10 pts)
        if   mv>160: sc_val += 10
        elif mv>100: sc_val += 7
        else:        pen    += 5

        scores[flower] = max(0.0, sc_val-pen)

    total = sum(scores.values())+1e-6
    norm  = {k:round(v/total*100,1) for k,v in scores.items()}
    return sorted(norm.items(), key=lambda x:-x[1])


# ── Region detection ───────────────────────────────────────────────────────────
def detect_flower_regions(img_bgr:np.ndarray) -> List[BoundingBox]:
    h,w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,50,60]),  np.array([180,255,255])),
        cv2.inRange(hsv, np.array([0,0,200]),  np.array([180,40,255])))
    green = cv2.inRange(hsv, np.array([30,40,40]), np.array([90,255,255]))
    mask  = cv2.bitwise_and(mask, cv2.bitwise_not(green))
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    clean = cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k),cv2.MORPH_OPEN,k)
    cnts,_ = cv2.findContours(clean,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes  = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if (w*h)*0.01 < area < (w*h)*0.95:
            rx,ry,rw,rh = cv2.boundingRect(cnt)
            pad=20
            boxes.append(BoundingBox(max(0,rx-pad),max(0,ry-pad),
                min(w-max(0,rx-pad),rw+pad*2),min(h-max(0,ry-pad),rh+pad*2)))
    boxes = _merge(boxes)
    if not boxes: boxes=[BoundingBox(0,0,w,h)]
    return sorted(boxes, key=lambda b:b.w*b.h, reverse=True)[:3]

def _merge(boxes, iou=0.3):
    if not boxes: return boxes
    merged=[]; used=[False]*len(boxes)
    for i,b1 in enumerate(boxes):
        if used[i]: continue
        x1,y1,w1,h1=b1.x,b1.y,b1.w,b1.h
        for j,b2 in enumerate(boxes):
            if i==j or used[j]: continue
            ox=max(0,min(x1+w1,b2.x+b2.w)-max(x1,b2.x))
            oy=max(0,min(y1+h1,b2.y+b2.h)-max(y1,b2.y))
            inter=ox*oy; union=w1*h1+b2.w*b2.h-inter
            if inter/(union+1e-6)>iou:
                x1=min(x1,b2.x);y1=min(y1,b2.y)
                x2=max(x1+w1,b2.x+b2.w);y2=max(y1+h1,b2.y+b2.h)
                w1=x2-x1;h1=y2-y1;used[j]=True
        merged.append(BoundingBox(x1,y1,w1,h1));used[i]=True
    return merged


# ── Draw boxes ────────────────────────────────────────────────────────────────
def draw_detections(img_bgr:np.ndarray, detections:List[Detection]) -> np.ndarray:
    out=img_bgr.copy(); h,w=out.shape[:2]
    for d in detections:
        b=d.bbox; c=d.color; lbl=f"{d.flower}  {d.confidence:.1f}%"
        th=max(2,int(min(w,h)*0.004))
        cv2.rectangle(out,(b.x,b.y),(b.x+b.w,b.y+b.h),c,th)
        cl=max(12,int(min(b.w,b.h)*0.12)); ct=th+1
        for p1,p2 in [((b.x,b.y),(b.x+cl,b.y)),((b.x,b.y),(b.x,b.y+cl)),
                      ((b.x+b.w,b.y),(b.x+b.w-cl,b.y)),((b.x+b.w,b.y),(b.x+b.w,b.y+cl)),
                      ((b.x,b.y+b.h),(b.x+cl,b.y+b.h)),((b.x,b.y+b.h),(b.x,b.y+b.h-cl)),
                      ((b.x+b.w,b.y+b.h),(b.x+b.w-cl,b.y+b.h)),((b.x+b.w,b.y+b.h),(b.x+b.w,b.y+b.h-cl))]:
            cv2.line(out,p1,p2,c,ct)
        font=cv2.FONT_HERSHEY_DUPLEX; fs=max(0.5,min(w,h)/900)
        (tw,fth),_=cv2.getTextSize(lbl,font,fs,1)
        lx=b.x; ly=max(b.y-fth-10,0)
        cv2.rectangle(out,(lx,ly),(lx+tw+12,ly+fth+10),c,-1)
        cv2.putText(out,lbl,(lx+6,ly+fth+4),font,fs,(255,255,255),1,cv2.LINE_AA)
    return out


# ── Main classify ──────────────────────────────────────────────────────────────
def classify_image(img_bytes:bytes) -> dict:
    import base64
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Could not decode image")
    h,w = img.shape[:2]
    if max(h,w)>800:
        s=800/max(h,w); img=cv2.resize(img,(int(w*s),int(h*s)))

    # ── Refresh caches (non-blocking — uses TTL)
    _load_correction_cache()
    _load_custom_profiles()

    # ── Priority 1: Correction cache
    override   = _correction_override(img)

    # ── Priority 2: ML model
    ml_preds   = predict_with_model(img)
    use_ml     = len(ml_preds)>0

    if override:
        model_label = f"✅ Correction cache ({len(_correction_cache.get(override[0],[]))} samples)"
    elif use_ml:
        model_label = f"ML — SVM+RF ({len(ml_preds)} classes)"
        print(f"[classify] ✅ ML → {ml_preds[0][0]} ({ml_preds[0][1]*100:.1f}%)")
    else:
        n_custom = len(_custom_profiles)
        model_label = f"Rule-based HSV ({len(BUILT_IN_PROFILES)} built-in + {n_custom} custom profiles)"
        print(f"[classify] ⚠️  HSV fallback | custom profiles: {list(_custom_profiles.keys())}")

    boxes = detect_flower_regions(img)
    detections = []
    for bbox in boxes:
        if override:
            top_lbl, top_conf = override
            top5 = [override]
        elif use_ml:
            top_lbl  = ml_preds[0][0]
            top_conf = round(ml_preds[0][1]*100,1)
            top5     = [(l,round(p*100,1)) for l,p in ml_preds[:5]]
        else:
            sc = _rulebased_scores(img, bbox)
            if not sc: continue
            top_lbl,top_conf = sc[0]; top5=sc[:5]

        m = _meta(top_lbl)
        detections.append(Detection(
            flower=top_lbl.title(), confidence=top_conf,
            bbox=bbox, icon=m["icon"], color=m["color"],
            all_predictions=top5))

    if not detections:
        bbox = BoundingBox(0,0,img.shape[1],img.shape[0])
        sc   = _rulebased_scores(img,bbox)
        if sc:
            top_lbl,top_conf=sc[0]; m=_meta(top_lbl)
            detections.append(Detection(
                flower=top_lbl.title(),confidence=top_conf,bbox=bbox,
                icon=m["icon"],color=m["color"],all_predictions=sc[:5]))

    ann = draw_detections(img, detections)
    _,buf = cv2.imencode(".jpg",ann,[cv2.IMWRITE_JPEG_QUALITY,92])
    best  = detections[0] if detections else None

    return {
        "annotated_image":  f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}",
        "model_used":       model_label,
        "custom_profiles":  list(_custom_profiles.keys()),
        "detections": [{
            "flower":     d.flower,
            "confidence": d.confidence,
            "icon":       d.icon,
            "bbox":       {"x":d.bbox.x,"y":d.bbox.y,"w":d.bbox.w,"h":d.bbox.h},
            "top5":       [{"flower":str(f).title(),"confidence":float(c)} for f,c in d.all_predictions],
        } for d in detections],
        "top_flower":     best.flower     if best else "Unknown",
        "top_confidence": best.confidence if best else 0.0,
        "top_icon":       best.icon       if best else "🌸",
    }