# 🎬 CelebMatch — Celebrity Look-Alike Neural Network

> A deep learning application that identifies which of 13 celebrities you most resemble, using a MobileNetV2 transfer learning model trained on the Celebrity Faces Dataset. Built with TensorFlow/Keras and deployed via a polished Streamlit interface.

---

## 🌟 Features

| Feature | Description |
|---|---|
| 📁 **Image Upload** | Upload any face photo (JPG / PNG) for prediction |
| 📷 **Live Webcam** | Capture your face directly from the browser |
| 🎯 **Real-time Prediction** | Top celebrity match with confidence percentage |
| 📊 **Probability Breakdown** | Full softmax probabilities for all 13 celebrities |
| 🧠 **Model Info Tab** | Architecture, training curves, epoch table, evaluation metrics |
| 🖼️ **Reference Gallery** | See the matched celebrity's reference photo |

---

## 🏗️ Project Structure

```
CelebrityLook_A_Like/
│
├── Celebrity Faces Dataset/          ← Training dataset
│   ├── Angelina Jolie/
│   │   ├── 001_fe3347c0.jpg
│   │   └── ... (100 images)
│   ├── Brad Pitt/
│   ├── Denzel Washington/
│   ├── Hugh Jackman/
│   ├── Jennifer Lawrence/
│   ├── Johnny Depp/
│   ├── Kate Winslet/
│   ├── Leonardo DiCaprio/
│   ├── Megan Fox/
│   ├── Natalie Portman/
│   ├── Nicole Kidman/
│   ├── Robert Downey Jr/
│   └── Sandra Bullock/
│
├── train_model.py                    ← CNN training script (run first)
├── app.py                            ← Streamlit UI application
├── requirements.txt                  ← Python dependencies
│
├── celebrity_model_best.h5           ← ⭐ Best model by val_accuracy (used by app)
├── celebrity_model.h5                ← Final model (last epoch)
└── model_info.json                   ← Saved class names, history & metrics
```

---

## 🧠 Neural Network Architecture

CelebMatch uses **Transfer Learning** with **MobileNetV2** as the backbone, pretrained on ImageNet (1.4M images). This is critical for small datasets — training a CNN from scratch on ~80 images/class yields only ~8–21% accuracy, while transfer learning achieves **64.6%** on the same data.

```
Input Image: 128 × 128 × 3
          │
          ▼
┌─────────────────────────────────────────┐
│  MobileNetV2 Backbone                   │
│  (Pretrained on ImageNet 1.4M images)   │
│                                         │
│  Phase 1: Fully Frozen                  │
│  Phase 2: Last 30 layers fine-tuned     │
└─────────────────────────────────────────┘
          │
          ▼
  GlobalAveragePooling2D
          │
          ▼
   BatchNormalization
          │
          ▼
   Dense(256) → ReLU → Dropout(0.4)
          │
          ▼
   Dense(128) → ReLU → Dropout(0.3)
          │
          ▼
   Dense(13) → Softmax
          │
          ▼
  Celebrity Prediction (0–12)
```

### Two-Phase Training Strategy

| Phase | Layers Active | Learning Rate | Epochs |
|---|---|---|---|
| Phase 1 — Head Only | Classification head only | 1e-3 | Up to 15 |
| Phase 2 — Fine-Tune | Head + last 30 MobileNetV2 layers | 1e-4 | Up to 15 |

**Smart Callbacks:**
- `ModelCheckpoint` — saves the best model by `val_accuracy` automatically  
- `EarlyStopping` — stops training if `val_loss` doesn't improve for 5 epochs  
- `ReduceLROnPlateau` — halves learning rate when `val_loss` plateaus (patience=3)

---

## 📊 Training Results

| Metric | Value |
|---|---|
| **Best Validation Accuracy** | **64.6%** |
| Best Epoch | Epoch 13 (Phase 1) |
| Final Validation Accuracy | 56.2% |
| Total Epochs Trained | 21 (early stopped) |
| Training Samples | 1,040 |
| Validation Samples | 260 |
| Classes | 13 celebrities |
| Model Size | ~25 MB |

> **Comparison:** Previous scratch CNN (3 Conv blocks) after 25 full epochs → **21% val accuracy**.  
> Transfer Learning (MobileNetV2) after 13 epochs → **64.6% val accuracy** — **3× better**.

---

## 🎭 Supported Celebrities

| # | Celebrity | # | Celebrity |
|---|---|---|---|
| 1 | Angelina Jolie | 8 | Leonardo DiCaprio |
| 2 | Brad Pitt | 9 | Megan Fox |
| 3 | Denzel Washington | 10 | Natalie Portman |
| 4 | Hugh Jackman | 11 | Nicole Kidman |
| 5 | Jennifer Lawrence | 12 | Robert Downey Jr |
| 6 | Johnny Depp | 13 | Sandra Bullock |
| 7 | Kate Winslet | — | — |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Anaconda (recommended) or standard Python environment

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
```
tensorflow>=2.10.0
numpy>=1.23.0
Pillow>=9.0.0
opencv-python>=4.6.0
streamlit>=1.22.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
```

### 2. Dataset Setup

Ensure the dataset is in the project root following this structure:
```
Celebrity Faces Dataset/
├── Angelina Jolie/   ← Folder name = class label
├── Brad Pitt/
└── ...
```

---

## 🚀 Usage

### Step 1 — Train the Model

> **Skip this step if `celebrity_model_best.h5` and `model_info.json` already exist.**

```bash
python train_model.py
```

This will:
- Load and augment the dataset (80/20 train/val split)
- Download MobileNetV2 ImageNet weights (first run only, ~14 MB)
- Run Phase 1: train classification head (up to 15 epochs)
- Run Phase 2: fine-tune last 30 MobileNetV2 layers (up to 15 epochs)
- Save `celebrity_model.h5` and `celebrity_model_best.h5`
- Save `model_info.json` with all metrics and training history

Expected output:
```
============================================================
  CelebMatch - Transfer Learning with MobileNetV2
============================================================
[OK] Classes (13): ['Angelina Jolie', 'Brad Pitt', ...]
     Training samples   : 1040
     Validation samples : 260

>> Phase 1 / 2 - Training classification head (15 epochs) ...
Epoch 1/15 – val_accuracy: 0.3731
...
>> Phase 2 / 2 - Fine-tuning MobileNetV2 last 30 layers ...
...
[OK] Final model saved -> celebrity_model.h5
[OK] Best model saved  -> celebrity_model_best.h5
[OK] Model info saved  -> model_info.json
```

### Step 2 — Launch the Web App

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🖥️ App Interface

### Tab 1 — 🔍 Find My Look-Alike

1. **Choose input method:** Upload Image or Use Webcam
2. **Upload / capture** your face photo
3. The model preprocesses the image (resize to 128×128, normalize to 0–1)
4. CNN predicts probabilities for all 13 celebrities
5. **Result displayed:**
   - Your photo ↔ matched celebrity reference photo
   - Celebrity name (gradient styled)
   - Confidence percentage (green = high, yellow = medium, red = low)
   - Expandable **Full Probability Breakdown** with progress bars for all 13

### Tab 2 — 🧠 Model & Neural Network Info

| Section | Content |
|---|---|
| Architecture | Layer-by-layer description of the model |
| Hyperparameters | Epochs, batch size, image size, optimizer |
| Evaluation Metrics | Val accuracy, val loss, best val accuracy |
| Training Curves | Line charts: Accuracy per Epoch, Loss per Epoch |
| History Table | Per-epoch train/val accuracy and loss |
| Celebrity Gallery | Reference images for all 13 celebrity classes |

---

## 🔄 How Prediction Works

```python
# 1. Preprocess
image = image.resize((128, 128)).convert("RGB")
arr   = numpy.array(image) / 255.0           # normalize to [0, 1]
arr   = numpy.expand_dims(arr, axis=0)        # add batch dimension → (1, 128, 128, 3)

# 2. Predict
predictions = model.predict(arr)              # shape: (1, 13) softmax probabilities

# 3. Map result
top_index   = numpy.argmax(predictions[0])   # index of highest probability
celebrity   = CLASS_NAMES[top_index]         # map index → folder name → celebrity name
confidence  = predictions[0][top_index]      # confidence score 0.0 – 1.0
```

---

## 📁 Output Files Explained

| File | Size | Description |
|---|---|---|
| `celebrity_model_best.h5` | ~25 MB | Best model checkpoint (highest val_accuracy) · used by app |
| `celebrity_model.h5` | ~25 MB | Final model after all epochs |
| `model_info.json` | <1 MB | JSON containing class names, training history (accuracy/loss per epoch), evaluation metrics, and all hyperparameters |

### `model_info.json` Schema

```json
{
  "model_params": {
    "epochs_trained": 21,
    "backbone": "MobileNetV2 (ImageNet pretrained)",
    "batch_size": 32,
    "img_size": [128, 128],
    "optimizer": "Adam",
    "lr_head": 0.001,
    "lr_finetune": 0.0001,
    "num_classes": 13,
    "architecture": ["..."]
  },
  "class_names": ["Angelina Jolie", "Brad Pitt", "..."],
  "evaluation": {
    "val_loss": 1.0789,
    "val_accuracy": 0.5615
  },
  "training_history": {
    "accuracy":     [0.1654, 0.2837, ...],
    "val_accuracy": [0.3731, 0.4192, ...],
    "loss":         [2.7644, 2.0588, ...],
    "val_loss":     [1.9189, 1.6860, ...]
  }
}
```

---

## 🔧 Data Augmentation

The training pipeline applies the following augmentations to artificially expand the small dataset:

| Augmentation | Value |
|---|---|
| Horizontal Flip | ✅ Enabled |
| Rotation | ± 15° |
| Brightness | 0.75× – 1.25× |
| Zoom | ± 15% |
| Width Shift | ± 10% |
| Height Shift | ± 10% |
| Shear | 5% |
| Rescale | 1 / 255 |

---

## 🧪 Why Not a Scratch CNN?

| Approach | Val Accuracy (25 epochs) | Notes |
|---|---|---|
| Scratch CNN (3 Conv blocks) | 21% | Insufficient data to learn meaningful features |
| **MobileNetV2 Transfer Learning** | **64.6%** | ImageNet features transfer well to face classification |

**Root cause:** With only ~80 training images per class, a scratch CNN overfits immediately and cannot generalize. MobileNetV2's pre-learned edge, texture, and shape detectors from 1.4M ImageNet images transfer directly to celebrity face recognition.

---

## 🔮 Future Enhancements

- [ ] **Siamese Network** — Compute facial similarity score directly instead of classification
- [ ] **Top-3 Matches** — Show top 3 celebrity look-alikes with confidence bars
- [ ] **Grad-CAM Visualization** — Highlight which face regions influenced the prediction
- [ ] **Face Detection Preprocessing** — Use MTCNN or MediaPipe to auto-crop and align faces before inference
- [ ] **Larger Dataset** — Add more celebrities and more images per class for higher accuracy
- [ ] **Mobile Deployment** — Convert to TensorFlow Lite for Android/iOS
- [ ] **Face Landmark Alignment** — Normalize face rotation/scale for better accuracy

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | TensorFlow 2.x / Keras |
| Model Backbone | MobileNetV2 (ImageNet) |
| Web UI | Streamlit |
| Image Processing | Pillow, OpenCV |
| Data Augmentation | Keras `ImageDataGenerator` |
| Styling | Custom CSS (glassmorphism dark theme) |
| Charts | Streamlit `st.line_chart` + Pandas |
| Serialization | JSON (training metrics), HDF5 (model weights) |

---

## 📄 License

This project is for educational and academic purposes. The Celebrity Faces Dataset is used under fair use for machine learning research.

---

<div align="center">
  <b>Built with TensorFlow · MobileNetV2 · Streamlit</b><br>
  <i>CelebMatch — Celebrity Look-Alike Neural Network</i>
</div>
