# CNN Models Directory
This directory is where your trained facial emotion recognition Convolutional Neural Network (CNN) model should be placed.

### Expected Model Specifications
- **Recommended filename**: `resnet50_emotion_best.keras`
- **Input dimension**: `(224, 224, 3)` RGB, preprocessed for ResNet50
- **Output Classes (FER2013 - 7 Classes in Alphabetical Order)**:
  1. `angry` (Index 0)
  2. `disgust` (Index 1)
  3. `fear` (Index 2)
  4. `happy` (Index 3)
  5. `neutral` (Index 4)
  6. `sad` (Index 5)
  7. `surprise` (Index 6)

---

The Flask service loads this model once at startup. To select a different model file, set `MODEL_FILENAME` in `backend/.env`:
```env
MODEL_FILENAME=resnet50_emotion_final.keras
```
Legacy 48x48 models are still supported and use grayscale normalization; ResNet50 models use RGB and the matching ResNet50 preprocessing.

---

### 🧪 Generating a Test Model:
If you want to test the entire backend pipeline before transferring your final trained weights, run:
```bash
python models/generate_dummy_model.py
```
This generates a valid legacy `emotion_model.h5` file with the original 48x48 CNN architecture for backend-pipeline smoke testing.
