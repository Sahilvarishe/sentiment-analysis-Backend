# ResNet50 Emotion Model

The scripts use the dataset already present in `backend/Dataset/train` and `backend/Dataset/test`. Class order is explicitly fixed as `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`. Grayscale files are loaded as RGB, resized to 224x224, and passed through the ResNet50 preprocessing function.

## Windows PowerShell Commands

Starting from the project root, run these commands. Python 3.11 must be installed, and the first training run needs internet access to download ImageNet weights unless they are already cached.

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python training/train_resnet50.py
python training/evaluate_model.py
python training/test_model.py ".\path\to\your\image.jpg"
python training/webcam_emotion.py
```

If PowerShell blocks environment activation, use `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` or activate from Command Prompt with `.venv\Scripts\activate.bat`. Evaluation is a separate step and reads only `Dataset/test`; the trainer uses an 80/20 split of `Dataset/train` and never loads the test split.

Training saves the best validation-accuracy model to `backend/models/resnet50_emotion_best.keras`, the final epoch model to `backend/models/resnet50_emotion_final.keras`, and accuracy/loss graphs under `backend/results`. Evaluation writes the classification report and named confusion matrix to that results directory. The webcam tool predicts only the largest face in each frame; all other detected faces are ignored for that frame.

To use the trained model in Flask, the backend defaults to `resnet50_emotion_best.keras` and loads it once when the emotion service starts. Set `MODEL_FILENAME=resnet50_emotion_final.keras` in `backend/.env` to select the final-epoch file instead. `focused` remains an optional recommendation mood and is not a facial-emotion output.

## Viva Explanation

ResNet50 is a 50-layer convolutional neural network. It was first trained on ImageNet, so its early layers already recognize general visual patterns such as edges, textures, and shapes. Residual (skip) connections pass information around groups of layers; they help gradients flow through a deep network and make it easier to fine-tune.

Each input image is made three-channel RGB and resized to 224x224. ResNet50's own preprocessing changes the pixel scale and channel means to the format used during ImageNet training. During training only, mild flips, rotations, zoom, translations, and contrast changes help reduce overfitting. The held-out test directory is not used for fitting or model selection.

The original ResNet50 classification top is removed (`include_top=False`). Its convolutional feature maps pass through Global Average Pooling, which summarizes each feature map without a large flattening layer. Dropout reduces co-adaptation, a 256-unit ReLU dense layer learns emotion-specific combinations, and a final seven-unit softmax layer produces probabilities in the fixed class order.

Training has two stages. First, all ImageNet ResNet50 layers are frozen and only the new head learns, using Adam at 0.001. Then the last 25% of the base layers (except Batch Normalization layers) are unfrozen and tuned at 0.00001 so useful pretrained features adapt without large weight changes. Early stopping watches validation loss, while the checkpoint keeps the best validation accuracy. Balanced class weights compensate for the uneven training counts without copying images.

Finally, the separate test split is evaluated once with loss, accuracy, weighted precision/recall/F1, a per-class classification report, and a confusion matrix. The output is a facial-expression estimate only; it does not diagnose a mental health condition. The backend maps the detected expression to a recommendation mood after prediction.