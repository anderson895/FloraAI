# 🌸 Flower Vision — cv2 BBox Classifier

## Architecture
```
frontend/   →  React + TypeScript (Vite)
backend/    →  Python Flask + cv2 + numpy
```

## How it works
1. User uploads a flower photo
2. **Frontend** sends image to Flask backend + uploads to Cloudinary
3. **Backend** (cv2):
   - Detects flower regions via HSV color segmentation + contour detection
   - Scores each region against 15 flower profiles (color histograms + HOG features)
   - Draws YOLOv8-style bounding boxes with labels
   - Returns annotated image (base64) + predictions
4. Frontend shows original vs annotated image toggle
5. Results + annotated image saved to **Firebase Firestore**

## Supported Flowers (15)
Sunflower · Rose · Daisy · Tulip · Lavender · Orchid · Dandelion · Hibiscus
Lily · Marigold · Poppy · Lotus · Chrysanthemum · Iris · Carnation

## Setup

### Backend
```bash
cd backend
pip install flask opencv-python numpy Pillow
python3 app.py
# → Running on http://localhost:5000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

## To improve accuracy
- Add more flower training images and extract features per species
- Replace HSV scoring with an ONNX model (export YOLOv8-cls to ONNX)
- The `classifier.py` is modular — swap out `score_against_profiles()` with any model


## RUles
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // ── Flower classifications (results ng detect) ──────────────────────────
    match /flower_classifications/{docId} {
      allow read: if true;
      allow create: if request.resource.data.keys().hasAll(['topFlower', 'confidence', 'imageUrl', 'annotatedUrl', 'timestamp'])
                   && request.resource.data.topFlower is string
                   && request.resource.data.confidence is number;
      allow update: if resource != null
                   && request.resource.data.topFlower is string
                   && request.resource.data.confidence is number;
      allow delete: if true;
    }
    // ── Training data (corrections + uploads) ───────────────────────────────
    match /flower_training_data/{docId} {
      allow read: if true;
      allow create: if request.resource.data.keys().hasAll(['label', 'imageUrl', 'timestamp'])
                   && request.resource.data.label is string
                   && request.resource.data.imageUrl is string;
      allow update, delete: if false;
    }
    // ── Lahat ng iba pa — block ──────────────────────────────────────────────
    match /{document=**} {
      allow read, write: if false;
    }
  }
}