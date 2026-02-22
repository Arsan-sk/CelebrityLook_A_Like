# 📖 CelebMatch — Complete Code Explanation
### Understanding `train_model.py` from Absolute Zero

> **Who is this for?** Someone who has never studied neural networks, machine learning, or computer vision. We explain every single concept from scratch — what a pixel is, how a computer "sees" an image, why we need neural networks, and exactly what every line of code does and *why* it exists.

---

## 🧠 Part 0 — Before We Read Any Code: What Is a Neural Network?

### What is an Image to a Computer?

When you look at a photo of Angelina Jolie, your brain instantly recognizes her. But a computer doesn't have eyes or a brain. To a computer, an image is just **a giant grid of numbers**.

Imagine a tiny 4×4 pixel image. Each pixel has a colour. Colours are represented as three numbers:
- **R** = Red intensity (0–255)
- **G** = Green intensity (0–255)  
- **B** = Blue intensity (0–255)

So a single red pixel = `[255, 0, 0]`. A white pixel = `[255, 255, 255]`. Black = `[0, 0, 0]`.

Our images are **128×128 pixels**, each with 3 colour channels. So each image is a **3D grid** of numbers:
```
Shape: (128 rows) × (128 columns) × (3 colour channels)
= 49,152 individual numbers per image
```

The computer's job is to look at 49,152 numbers and decide: "This is probably Angelina Jolie."

---

### What is a Neural Network?

A neural network is a **mathematical function** inspired by how neurons in the human brain connect. It takes numbers in (the pixel values) and produces numbers out (probabilities for each celebrity).

Think of it like a factory assembly line:

```
Raw Image Numbers
      ↓
[Detector Layer 1] → finds edges (where light meets dark)
      ↓
[Detector Layer 2] → combines edges into shapes (eyes, nose outlines)
      ↓
[Detector Layer 3] → combines shapes into face parts (an eye shape, a jawline)
      ↓
[Detector Layer 4] → combines face parts into identity features
      ↓
[Decision Layer]   → "Based on those features, this is 64% Natalie Portman"
```

Each "Detector Layer" is a `Conv2D` (Convolutional) layer. Each "Decision Layer" is a `Dense` layer. This is exactly what the code builds.

---

### What is "Training"?

Training is how the network **learns**. It starts completely random — like a newborn baby who has never seen a face. Then:

1. Show it an image of Brad Pitt
2. It makes a (wrong) guess — "I think this is Sandra Bullock, 8%"
3. It knows the right answer is Brad Pitt
4. It **adjusts its internal numbers** (called weights) very slightly to make "Brad Pitt" more likely next time
5. Repeat this **32,500 times** (1040 images × 25 passes through the data)
6. After thousands of corrections, the network has "learned" what each celebrity looks like

This adjustment process is called **backpropagation**, and the rule for how much to adjust is controlled by the **optimizer** (Adam) and **loss function** (categorical cross-entropy).

---

## 📄 Part 1 — The File Header (Lines 1–18)

```python
# -*- coding: utf-8 -*-
```

**What this is:** A special comment at the very top of a Python file.

**Why it exists:** Python files are just text. Text is stored as bytes in memory. Different countries use different systems to map bytes to characters (ASCII, Latin-1, UTF-8, etc.). The `# -*- coding: utf-8 -*-` line tells Python: "When you read this file, treat the text as UTF-8 encoding." 

UTF-8 is a universal standard that supports every language character — English, Arabic, Hindi, emoji, etc. Without this, printing characters like `→` or `✔` on Windows can crash the program.

```python
"""
CelebMatch - Celebrity Look-Alike CNN  (Transfer Learning with MobileNetV2)
Training Script
...
"""
```

**What this is:** A triple-quoted string called a **docstring**. It's a multi-line comment.

**Why it exists:** This is documentation for any human reading the file. Python completely ignores everything between `"""..."""`. It's there to explain the file's purpose in plain English before any code begins.

---

## 📦 Part 2 — Importing Libraries (Lines 20–34)

```python
import os
import sys
import json
import numpy as np
import tensorflow as tf
```

**What is "importing"?**  
Python by itself is quite small. It doesn't know how to do matrix math, load images, or build neural networks. Libraries (also called packages or modules) are pre-written code written by experts that we can "borrow." `import` brings them into our script.

Think of imports like opening a toolbox. You don't build your own hammer — you open the toolbox and take one out.

**`import os`**  
`os` = Operating System. This library lets Python talk to Windows/Mac/Linux. We use it to check if folders exist (`os.path.isdir`) and list files in folders (`os.listdir`). Without this, Python can't navigate your file system.

**`import sys`**  
`sys` = System. This library talks to the Python interpreter itself. We use it to check and change the text encoding of the output console — critical on Windows where the default encoding (cp1252) cannot display special characters.

**`import json`**  
JSON = JavaScript Object Notation. It's a universal text format for storing structured data (like Python dictionaries) in a file that any programming language can read. We use it to save the training results (accuracy, loss per epoch) into `model_info.json`.

**`import numpy as np`**  
NumPy = Numerical Python. This is the foundation of all scientific computing in Python. It provides **arrays** — fast, efficient grids of numbers. `as np` means we can type `np.array(...)` instead of `numpy.array(...)` — just a shortcut.

Why do we need NumPy for images? Because images are arrays! A 128×128×3 image becomes a NumPy array of shape `(128, 128, 3)`. NumPy can do math on millions of numbers at once using optimized C code under the hood — far faster than plain Python lists.

**`import tensorflow as tf`**  
TensorFlow is Google's machine learning framework. It provides:
- Tensor operations (tensors are just multi-dimensional arrays like NumPy arrays, but can run on GPU)
- Automatic differentiation (the math behind learning from mistakes — backpropagation)
- High-level neural network building blocks via its sub-library Keras

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

**`from X import Y`** means: "From the X toolbox, take only the Y tool."

`ImageDataGenerator` is a class that:
1. Loads images from folders automatically
2. Applies random transformations (augmentation) at training time to create variety
3. Feeds batches of images to the model during training

Without this, you'd have to manually write code to open every image file, resize it, normalize it, and batch it.

```python
from tensorflow.keras.applications import MobileNetV2
```

`MobileNetV2` is a pre-built, pre-trained neural network model. "Pre-trained" means someone already trained it on 1.4 million images (the ImageNet dataset) for weeks. We download the knowledge it learned and reuse it for our celebrity task. This is **Transfer Learning** — explained in depth later.

```python
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
```

These are the **building blocks** of neural networks. Each is a type of layer:
- `GlobalAveragePooling2D` — compresses feature maps into a flat vector
- `Dense` — a fully connected layer where every neuron connects to every previous neuron
- `Dropout` — randomly turns off some neurons during training to prevent overfitting
- `BatchNormalization` — normalizes the outputs of a layer to stabilize training

Each is explained in full detail when we use them.

```python
from tensorflow.keras.models import Model
```

`Model` is the container that holds all the layers together. You define inputs, outputs, and layers — `Model` stitches them into one object you can train, save, and use for predictions.

```python
from tensorflow.keras.optimizers import Adam
```

`Adam` is the **optimizer** — the algorithm that updates the model's internal numbers (weights) after each mistake. Adam stands for "Adaptive Moment Estimation." It's one of the best general-purpose optimizers because it adapts the learning rate for each individual weight automatically.

```python
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
```

**Callbacks** are functions that run automatically at specific moments during training (end of each epoch, end of each batch). Like a supervisor watching the training and stepping in when needed.
- `ModelCheckpoint` — saves the model to disk whenever it improves
- `EarlyStopping` — stops training automatically when the model stops improving
- `ReduceLROnPlateau` — reduces the learning rate when the model is stuck

---

## 🔧 Part 3 — UTF-8 Fix for Windows (Lines 36–38)

```python
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
```

**What is `sys.stdout`?**  
`stdout` = "standard output" = the terminal/console where Python prints text. When you run `python train_model.py`, everything printed by `print(...)` goes to `stdout`.

**Why this fix is needed:**  
Windows terminals by default use "cp1252" encoding (Code Page 1252), a legacy Western European encoding that cannot display special symbols like `→`, `✔`, or arrows. If you try to print these characters on a Windows terminal without this fix, Python throws a `UnicodeEncodeError` and the program crashes.

**What the fix does:**  
`sys.stdout.reconfigure(encoding='utf-8')` tells Python to switch the console's output mode to UTF-8, which supports every character. The `if` condition checks first — if it's already UTF-8 (as on Linux/Mac), we don't need to do anything.

---

## ⚙️ Part 4 — Configuration Constants (Lines 40–50)

```python
DATASET_PATH = "Celebrity Faces Dataset"
```

**What is a constant?**  
A constant is a variable whose value we set once and don't change. Python has no enforcement mechanism for constants — it's just a convention to write them in ALL_CAPS to signal "don't change this accidentally."

**Why this is here:** Instead of typing `"Celebrity Faces Dataset"` 5 times in the code, we store it once in a variable. If we ever rename the folder, we only change one line here instead of hunting through 5 places.

```python
IMG_SIZE = (128, 128)
```

Every image fed to the neural network must be **exactly the same size**. Neural networks have fixed-size inputs — you can't have some images be 200×300 and others 500×600. We resize every image to 128×128 pixels before training. A tuple `(128, 128)` means (width=128, height=128).

**Why 128×128?**  
It's a balance between:
- **Too small** (e.g., 32×32): Too few pixels, facial details are lost, model can't distinguish faces well
- **Too large** (e.g., 512×512): Very slow training, uses huge amounts of GPU memory
128×128 captures enough facial detail while keeping training time reasonable on a CPU.

```python
BATCH_SIZE = 32
```

**What is a batch?**  
Instead of showing the model one image at a time (too slow) or all 1040 images at once (too much memory), we show it **32 images at a time**. This group of 32 is called a "batch" or "mini-batch."

After processing each batch, the model makes 32 predictions, compares them all to correct answers, calculates an average error, and updates its weights once.

One full pass through all training data = one **epoch**. With 1040 training images and batch size 32:
```
1040 / 32 = 32.5 → 33 batches per epoch
```

**Why 32?** It's empirically shown to work well — large enough to give a stable gradient estimate, small enough to fit in memory and require frequent weight updates (which helps learning).

```python
EPOCHS_HEAD  = 15    # Phase 1: train only the top head
EPOCHS_FINE  = 15    # Phase 2: fine-tune last 30 layers of MobileNetV2
```

An **epoch** = one complete pass through all training images. With 1040 images and batch size 32, training for 15 epochs means the model sees each training image 15 times.

We train in two phases (explained fully in Part 8):
- Phase 1 (15 epochs): Only our custom top layers train. MobileNetV2 stays frozen.
- Phase 2 (15 epochs): We also unfreeze part of MobileNetV2 for fine-tuning.

```python
LR_HEAD      = 1e-3    # = 0.001
LR_FINE      = 1e-4    # = 0.0001
```

**What is Learning Rate?**  
The learning rate controls **how big a step** the optimizer takes when adjusting weights after each mistake.

Imagine you're blindfolded on a hilly landscape, trying to reach the lowest valley (lowest error). The learning rate determines how far you step in any direction:
- **Too high** (e.g., 0.1): You take huge steps and jump over the valley, bouncing around never settling
- **Too low** (e.g., 0.000001): You take tiny steps and training takes forever
- **Just right** (0.001): You step confidently and eventually settle at the bottom

We use a **smaller** learning rate in Phase 2 (1e-4 vs 1e-3) because:
- In Phase 2, we're slightly modifying weights the MobileNetV2 already learned over weeks of ImageNet training
- Aggressive updates (high LR) would destroy those carefully learned weights
- A gentle rate (10× smaller) fine-tunes without forgetting

```python
BEST_MODEL   = "celebrity_model_best.h5"
FINAL_MODEL  = "celebrity_model.h5"
INFO_FILE    = "model_info.json"
```

Filenames where we save the results. `.h5` is the **HDF5 format** — a binary file format that stores the entire trained model: its architecture (layer structure) and weights (1.4 million learned numbers). Think of it as a "save game" for the neural network.

---

## 🖼️ Part 5 — Data Generators & Augmentation (Lines 52–92)

### Understanding Image Augmentation

We have only ~80 training images per celebrity. That's tiny. A human child learns to recognize faces from millions of real-life sightings. How do we help our model generalize?

**Data Augmentation** = artificially create more variety from existing images, so the model sees more "versions" of each face and doesn't memorize the exact training photos.

Think of it like this: You have one photo of Brad Pitt. You:
- Flip it horizontally (mirror image)
- Rotate it slightly
- Zoom in a little
- Adjust the brightness

Now you have 4 versions of the same photo. The model trains on all of these, learning "even rotated, even mirrored, even darker — it's still Brad Pitt."

```python
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    brightness_range=[0.75, 1.25],
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
)
```

**`rescale=1.0 / 255`**

Pixel values range from 0 to 255. Neural networks work much better when inputs are **small numbers** between 0 and 1. Dividing every pixel by 255 converts the range:
```
0 → 0.0   (completely black pixel)  
128 → 0.502  (mid-grey pixel)
255 → 1.0   (completely white pixel)
```

This is called **normalization**. Without it, early in training the model receives enormous numbers (like 240) and makes extreme prediction errors, causing the weight updates to be wildly unstable (exploding or vanishing gradients).

**`validation_split=0.2`**

We want to test the model on images it has **never seen during training**. Otherwise, we can't know if the model has genuinely learned faces or just memorized the exact training photos.

`0.2` = 20% of images are reserved purely for validation (testing). 80% train, 20% validate.
```
Total: 1300 images
Training: 1040 images (80%)
Validation: 260 images (20%)
```

**`horizontal_flip=True`**

Randomly flips an image left-to-right (mirror). A face looks the same mirrored — it's still the same person. This doubles the effective variety.

**`rotation_range=15`**

Randomly rotates the image by any angle from -15° to +15°. Real photos aren't always perfectly upright — people tilt their heads, cameras tilt. Teaching the model on tilted images makes it robust.

**`brightness_range=[0.75, 1.25]`**

Randomly makes the image 25% darker or 25% brighter. Simulates different lighting conditions — indoor dim lighting, direct sunlight, flash photography.

**`zoom_range=0.15`**

Randomly zooms into the image by up to 15%. Simulates photos taken at different distances — close-up head shot vs. photo taken from farther away.

**`width_shift_range=0.1` and `height_shift_range=0.1`**

Randomly shifts the image horizontally or vertically by up to 10% of its size. Simulates off-center framing — the face isn't always perfectly centered in the photo.

**`shear_range=0.05`**

Applies a subtle "shearing" transformation — like tilting the image sideways without rotating it. Creates distorted perspective effects. Rare in real photos but improves robustness.

```python
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)
```

**Why does the validation generator have NO augmentation?**  

This is crucial. Augmentation is only applied to training images. The validation set must represent **real-world images** that the model will encounter when deployed. If we augment validation images too, we'd be testing the model on distorted versions of faces — that's not a fair measure of real-world performance.

The only thing we do to validation images is `rescale` (normalize to 0–1), because that's necessary for the model to process them.

```python
train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)
```

**`flow_from_directory(DATASET_PATH)`**  
Reads the folder structure and automatically:
- Scans `Celebrity Faces Dataset/` for subfolders
- Each subfolder name → one class (e.g., `"Angelina Jolie"` = class 0, `"Brad Pitt"` = class 1, etc., sorted alphabetically)
- Loads images from all subfolders

**`target_size=IMG_SIZE`**  
Every image is resized to `(128, 128)` pixels. This uses OpenCV interpolation — it either "zooms in" or "shrinks" each image to exactly 128×128, regardless of original size. This is done using **bilinear interpolation** (averaging nearby pixels to compute new pixel values).

**`class_mode="categorical"`**  
Tells the generator how to format the labels. With 13 celebrities, instead of labelling a Brad Pitt image as `1`, we use a **one-hot encoding**:

```
Angelina Jolie  → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Brad Pitt       → [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Denzel Washington→[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
...
Sandra Bullock  → [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

This is a vector of 13 numbers, all 0 except one 1 at the correct class position. Categorical cross-entropy loss compares this 13-number vector to the model's 13-number prediction to calculate how wrong the model is.

**`subset="training"`**  
Uses the 80% training split (not the 20% validation split).

**`shuffle=True`**  
Randomly shuffles the order images are fed to the model each epoch. Why? If all Angelina Jolie images are fed first, then all Brad Pitt images, the model "forgets" about Angelina by the time it updates weights on Brad Pitt's images. Shuffling ensures every batch contains a mix of celebrities.

**`seed=42`**  
A random seed ensures reproducibility. With `seed=42`, the same images always go to train vs. validation, and shuffling is done the same way. The number 42 has no special meaning — it's a pop culture reference (The Hitchhiker's Guide to the Galaxy). Any number works; what matters is consistency.

---

## 🏗️ Part 6 — Building the Neural Network (Lines 101–120)

### 🔬 Deep Dive: What Is a Convolution? (The Core of CNN)

Before reading the code, you must understand what a **convolutional layer** does.

Imagine you have a 5×5 image:
```
10  20  30  40  50
11  21  31  41  51
12  22  32  42  52
13  23  33  43  53
14  24  34  44  54
```

A **filter** (also called a kernel) is a small grid, say 3×3:
```
 0  -1   0
-1   4  -1
 0  -1   0
```

The convolution operation slides this 3×3 filter across the image, one step at a time. At each position, it does an **element-wise multiplication and sum**:

```
Position (0,0): The filter sits on top of:
10  20  30
11  21  31
12  22  32

Multiply each element:
(10×0) + (20×-1) + (30×0) +
(11×-1) + (21×4) + (31×-1) +
(12×0) + (22×-1) + (32×0)
= 0 + (-20) + 0 + (-11) + 84 + (-31) + 0 + (-22) + 0
= 0
```

The result `0` is written to the output. The filter slides one pixel right, repeats. One slide down, repeats. And so on.

**Why is this powerful?** Different filters detect different features:
- A filter with sharp differences detects **edges** (where colour changes abruptly — like the outline of a nose)
- A filter with circular pattern detects **circles** (like pupils/iris)
- A filter that looks for diagonal lines detects **eyebrows**

In a neural network, the values in the filter are **learned weights**. Instead of us hand-designing filters, the network learns on its own which filter patterns are most useful for recognizing celebrity faces.

With 32 filters in the first layer, the network learns 32 different types of features simultaneously.

### What is "Padding = Same"?

When a 3×3 filter slides over a 5×5 image, the output is only 3×3 (the filter can't go beyond the edges). We lose spatial information at the borders.

**`padding="same"`** adds a border of zeros around the image before the filter slides:
```
0   0   0   0   0   0   0
0  10  20  30  40  50   0
0  11  21  31  41  51   0
0  12  22  32  42  52   0
0  13  23  33  43  53   0
0  14  24  34  44  54   0
0   0   0   0   0   0   0
```

Now the 3×3 filter has room to center on edge pixels too. The output size = input size ("same"). This preserves spatial dimensions through layers.

---

```python
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False
```

**`MobileNetV2`** is a neural network architecture designed by Google in 2018. It has 53 layers of convolutions, batch normalizations, and activation functions, arranged in an efficient "inverted residual" pattern that gives excellent accuracy with minimal computation.

**`input_shape=(128, 128, 3)`**  
Tells MobileNetV2 to expect images of exactly 128 rows × 128 columns × 3 colour channels (RGB). The model's first convolution layer is built to match this shape.

**`include_top=False`**  
MobileNetV2 was originally trained to classify 1000 types of objects (dogs, cars, boats, etc. from ImageNet). Its last few layers are "classification layers" for those 1000 classes.

`include_top=False` removes those last 1000-class classification layers. We only take the "feature extractor" part — all the convolutional layers that learned to detect shapes, edges, textures — and we'll add our own 13-celebrity classification layers on top.

**`weights="imagenet"`**  
Downloads and loads the weights (1.4 million learned numbers) from the model that was trained on ImageNet for weeks on powerful GPUs. This is the core of **Transfer Learning** — we borrow someone else's years of training.

**`base_model.trainable = False`**  
Freezes all the MobileNetV2 layers. "Frozen" means their weights **will NOT change** during Phase 1 training. They're read-only.

Why freeze? If we immediately let MobileNetV2's weights update with our tiny celebrity dataset, the carefully learned ImageNet features would be destroyed ("catastrophic forgetting"). We first train only our new custom top layers until they're stable, then gradually unfreeze MobileNetV2 in Phase 2.

---

```python
x = base_model.output
x = GlobalAveragePooling2D()(x)
```

**`base_model.output`** is the output tensor from MobileNetV2's last convolutional layer. Its shape is `(4, 4, 1280)` — a 4×4 spatial grid with 1280 feature channels (1280 different learned feature detectors, each producing a 4×4 map).

**`GlobalAveragePooling2D`** takes this `(4, 4, 1280)` volume and **reduces it to a flat 1280-number vector** by averaging each 4×4 feature map into a single number.

For each of 1280 feature maps:
```
Average of all 4×4 = 16 numbers → 1 number
Result: 1280 numbers total (a 1D vector)
```

**Why do this?** The Dense (fully connected) layers that follow need a flat 1D vector as input, not a 3D volume. GAP is a smooth way to transition from "spatial feature maps" to "flat description vector."

```python
x = BatchNormalization()(x)
```

**BatchNormalization** normalizes the values in the 1280-number vector across the current batch of 32 images. It adjusts the values to have:
- **Mean ≈ 0** (centered around zero)
- **Standard deviation ≈ 1** (similar scale)

**Why does this help?** Consider what happens without it: Layer 1 outputs numbers between 0 and 1000. Layer 2 tries to learn with inputs of 0–1000, but it was initialized expecting 0–1. The learning becomes unstable and very slow. BatchNormalization keeps the "scale" of numbers consistent throughout the network, making training much faster and more stable.

```python
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
```

**`Dense(256)`** — a fully connected layer with 256 neurons (also called nodes or units).

**What is a neuron?** It receives the 1280 inputs, multiplies each by a learned weight, sums them all up, and adds a "bias" number:
```
output = (input_1 × weight_1) + (input_2 × weight_2) + ... + (input_1280 × weight_1280) + bias
```

With 256 neurons and 1280 inputs, this layer has:
```
1280 × 256 weights + 256 biases = 327,936 learnable parameters
```

This layer can learn complex combinations of the 1280 MobileNetV2 features.

**`activation="relu"`** — ReLU = Rectified Linear Unit. It's an activation function applied after every neuron's sum:
```
ReLU(x) = max(0, x)
  - If the sum is positive → keep it as-is
  - If the sum is negative → output 0
```

**Why do we need activation functions?**  
Without them, stacking Dense layers is mathematically identical to having just one Dense layer — no matter how many layers you add, the combined effect is still just a linear transformation (y = Wx + b). Activation functions introduce **non-linearity**, allowing the network to learn complex, curved decision boundaries rather than just straight lines.

ReLU is simple and fast to compute, and doesn't suffer from "vanishing gradients" (a problem where learning signals become too small to be useful in deep networks).

**`Dropout(0.4)`** — randomly sets 40% of the 256 neuron outputs to zero during each training step.

**Why randomly destroy information?** It prevents **overfitting**. When neurons "rely on each other" (neuron A always works with neuron B), removing A randomly forces B to learn to work independently. This produces a more **robust, generalized** model that works on new faces it hasn't seen.

During prediction (inference), Dropout is automatically disabled — all 256 neurons work together.

```python
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
```

A second Dense layer, smaller (128 neurons). We're funneling the 256 abstracted features into 128 high-level discriminating features. Less dropout (30%) because at this deeper level, the features are more refined and we want to preserve more of them.

```python
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
```

**`Dense(NUM_CLASSES)`** = `Dense(13)` — the final output layer. 13 neurons, one for each celebrity.

**`activation="softmax"`** — This is the critical final activation. Softmax takes the 13 raw output numbers and converts them into **probabilities that sum to exactly 1.0**:

```
Raw outputs (logits):  [2.1, 0.3, -1.2, 4.5, 0.8, -0.3, 1.1, 3.2, -0.9, 1.7, 0.6, 2.9, -0.4]
After Softmax:         [0.05, 0.01, 0.00, 0.35, 0.01, 0.01, 0.02, 0.09, 0.00, 0.04, 0.01, 0.12, 0.00]
Sum:                   = 1.00 (exactly)
```

The celebrity with the highest probability is the prediction. In this example, class index 3 (Denzel Washington) has 35% probability — the highest.

```python
model = Model(inputs=base_model.input, outputs=outputs, name="CelebMatch_MobileNetV2")
```

`Model(inputs=..., outputs=...)` connects everything into one model object:
- **Input:** A 128×128×3 image
- **Output:** A 13-number probability vector
- All layers in between are automatically tracked

`name="CelebMatch_MobileNetV2"` is just a label for identification (appears in `model.summary()`).

---

## 🎛️ Part 7 — Callbacks: Auto-Supervisors (Lines 123–145)

```python
def make_callbacks():
    return [
        ModelCheckpoint(...),
        EarlyStopping(...),
        ReduceLROnPlateau(...),
    ]
```

**`def make_callbacks()`** defines a function. Functions are reusable blocks of code. We defined this as a function (rather than creating callbacks directly) because we need callbacks **twice** — once for Phase 1, once for Phase 2. Calling `make_callbacks()` creates a fresh set each time.

```python
ModelCheckpoint(
    BEST_MODEL,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
),
```

**`ModelCheckpoint`** watches the training and saves the model to disk automatically.

- **`BEST_MODEL`** = `"celebrity_model_best.h5"` — the filename to save to
- **`monitor="val_accuracy"`** — watch the validation accuracy after each epoch
- **`save_best_only=True`** — only overwrite the file if this epoch's val_accuracy is better than all previous epochs

Without this, if training peaks at epoch 11 (64% accuracy) but then gets slightly worse at epoch 12 (63%), we'd lose the best version. With `save_best_only`, the best is always preserved.

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
),
```

**`EarlyStopping`** stops training automatically when the model stops getting better.

- **`monitor="val_loss"`** — watch the validation loss after each epoch
- **`patience=5`** — allow 5 consecutive epochs of no improvement before stopping
- **`restore_best_weights=True`** — when stopping, roll back the model weights to the best epoch (not the last one)

Without early stopping, training would continue all 15 epochs even if the model stopped improving at epoch 8, wasting time and potentially making the model worse (overfitting).

```python
ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1,
),
```

**`ReduceLROnPlateau`** reduces the learning rate when training is stuck.

- **`factor=0.5`** — multiply learning rate by 0.5 (halve it) when triggered
- **`patience=3`** — trigger after 3 epochs of no val_loss improvement
- **`min_lr=1e-7`** — never go below this learning rate (preventing it from becoming uselessly tiny)

**Intuition:** If you're taking big steps and keep jumping over the valley bottom, take smaller steps to "land" more precisely.

---

## 🎓 Part 8 — Phase 1: Training the Head (Lines 147–160)

```python
model.compile(
    optimizer=Adam(learning_rate=LR_HEAD),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

**`model.compile`** configures the learning process — it connects the neural network to the mathematical machinery of learning.

**`optimizer=Adam(learning_rate=0.001)`**  
Adam is the algorithm that updates weights. After each batch, it:
1. Calculates how wrong the prediction was (loss)
2. Computes **gradients** — using calculus (chain rule / backpropagation), it figures out: "If I increase weight W by a tiny amount, does the error go up or down? By how much?"
3. Updates each weight in the direction that reduces error, by a step proportional to the learning rate

Adam is special because it maintains separate adaptive learning rates for each weight and uses "momentum" (memory of past gradients) to navigate more intelligently.

**`loss="categorical_crossentropy"`**  
The **loss function** measures how wrong the model's prediction is. With 13 celebrities and softmax probabilities:

```
Correct answer: Brad Pitt → [0,1,0,0,0,0,0,0,0,0,0,0,0]
Model says:               → [0.01, 0.03, 0.02, ...]
Cross-entropy = -log(0.03)  ← punishes low probability for correct class
             = -log(1.0) when perfect prediction = 0 (no error)
```

Cross-entropy heavily penalizes confident wrong answers and barely penalizes near-correct answers. This shapes the model to be calibrated and confident about correct answers.

**`metrics=["accuracy"]`**  
`accuracy` = percentage of images classified correctly. This is what we as humans care about — "how often is the model right?" It's not used for learning (that's what loss is for), just for reporting during training.

```python
history_head = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_HEAD,
    callbacks=make_callbacks(),
    verbose=1,
)
```

**`model.fit`** is the actual training. This single function call runs the entire training loop:

For each epoch (1 to 15):
  - For each batch (1 to 33 batches of 32 images):
    1. Load 32 images + apply augmentation
    2. Pass images through all layers → get 32 predictions
    3. Compare predictions to correct labels → compute loss
    4. Run backpropagation → compute gradients
    5. Adam updates all trainable weights
  - After all 33 batches: evaluate on the 260 validation images (no augmentation)
  - Print epoch summary: train loss, train accuracy, val loss, val accuracy
  - Run callbacks (save best model? stop early? reduce LR?)

**`verbose=1`** — print a progress bar for every batch during each epoch. `verbose=0` would train silently; `verbose=2` would print one line per epoch.

---

## 🔓 Part 9 — Phase 2: Fine-Tuning (Lines 162–180)

```python
for layer in base_model.layers[-30:]:
    layer.trainable = True
```

**`base_model.layers`** is a Python list of all layers in MobileNetV2 (153 layers total).

**`[-30:]`** — Python list slicing. Negative index counts from the end. `-30:` means "from the 30th-from-last layer to the last layer." So we're selecting the **last 30 layers** of MobileNetV2.

**Setting `.trainable = True`** unfreezes those 30 layers — their weights can now change during training.

**Why only the last 30 (not all 153)?**  
Neural networks learn **hierarchical features**:
- **Early layers** (layers 1–50): Learn basic, universal features — edges, gradients, colour blobs. These are the same for any image of anything. We keep these frozen.
- **Later layers** (layers 120–153): Learn high-level, task-specific features — complex shapes, object parts. These benefit from being adapted to celebrity faces specifically.

By unfreezing only the last 30, we fine-tune high-level features while preserving universal low-level features.

```python
model.compile(
    optimizer=Adam(learning_rate=LR_FINE),   # 0.0001 — 10× smaller
    ...
)
```

We must **recompile** the model after changing which layers are trainable. Keras builds a new computation graph that includes the newly unfrozen layers.

The 10× smaller learning rate (1e-4 vs 1e-3) prevents destroying MobileNetV2's carefully learned weights.

```python
history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_HEAD + EPOCHS_FINE,    # = 30
    initial_epoch=EPOCHS_HEAD,           # = 15
    ...
)
```

**`epochs=30, initial_epoch=15`** — Keras counts epochs from `initial_epoch`. So this runs epochs 16, 17, ..., 30. It continues the training from where Phase 1 left off, without resetting anything.

---

## 💾 Part 10 — Saving the Model (Lines 182–185)

```python
model.save(FINAL_MODEL)
```

`model.save("celebrity_model.h5")` serializes the entire model into a single HDF5 file:
- The architecture (which layers in what order, with what configuration)
- All weight arrays (1.4M numbers)
- The optimizer state (Adam's momentum values for each weight)

This allows us to later do `tf.keras.models.load_model("celebrity_model.h5")` and get the **exact same model** without re-training.

---

## 📊 Part 11 — Merging Training Histories (Lines 187–196)

```python
def merge_hist(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

combined = {
    "accuracy":     merge_hist(history_head, history_fine, "accuracy"),
    "val_accuracy": merge_hist(history_head, history_fine, "val_accuracy"),
    "loss":         merge_hist(history_head, history_fine, "loss"),
    "val_loss":     merge_hist(history_head, history_fine, "val_loss"),
}
```

`history_head.history` and `history_fine.history` are Python dictionaries containing lists of metric values per epoch:
```python
history_head.history = {
    "accuracy":     [0.165, 0.284, 0.352, ...],  # one value per epoch
    "val_accuracy": [0.373, 0.419, 0.487, ...],
    "loss":         [2.764, 2.059, 1.815, ...],
    "val_loss":     [1.919, 1.686, 1.487, ...],
}
```

The `merge_hist` function concatenates Phase 1 and Phase 2 lists using `+` (Python list concatenation):
```python
[epoch1, epoch2, ..., epoch15] + [epoch16, epoch17, ..., epoch21]
= [epoch1, epoch2, ..., epoch21]
```

This gives us a continuous training history across all 21 epochs, which we can plot as a smooth accuracy/loss curve in the Streamlit app.

---

## 📝 Part 12 — Evaluation & Saving model_info.json (Lines 198–246)

```python
val_loss, val_acc = model.evaluate(val_data, verbose=1)
```

`model.evaluate` runs the model on all 260 validation images WITHOUT augmentation, WITHOUT training (weights don't change). It returns the average loss and accuracy across all validation images. This is the official "final score" of the trained model.

```python
model_info = {
    "model_params": { ... },
    "class_names": CLASS_NAMES,
    "evaluation": { "val_loss": ..., "val_accuracy": ... },
    "training_history": { "accuracy": [...], "val_accuracy": [...], ... },
}
```

This dictionary contains everything needed to display model information in the Streamlit app without having to re-train:
- **`class_names`**: The sorted list of celebrity names, used to map prediction index → celebrity name
- **`evaluation`**: Final test scores
- **`training_history`**: Per-epoch metrics to draw the accuracy/loss charts
- **`model_params`**: Hyperparameters used — so the app can display "trained with batch size 32, Adam optimizer, etc."

```python
with open(INFO_FILE, "w", encoding="utf-8") as f:
    json.dump(model_info, f, indent=2)
```

**`with open(...) as f`** — Python's context manager for file I/O. It safely opens the file, gives us `f` (a file handle), and automatically closes the file when done (even if an error occurs). Forgetting to close files can cause data corruption or locks.

**`json.dump(model_info, f, indent=2)`** — converts the Python dictionary into a JSON-formatted string and writes it to the file. `indent=2` makes the JSON file human-readable with 2-space indentation (rather than one long line).

**`encoding="utf-8"`** — ensures the file is written in UTF-8, so it can be read correctly on any operating system.

---

## 🔄 The Complete Training Loop — Visualized

```
Start
  │
  ▼
Load 13 celebrity folders → 1300 images
  │
  ▼
Split: 80% train (1040) | 20% val (260)
  │
  ▼
Build Model:
  MobileNetV2 (frozen) → GAP → BN → Dense(256) → Dense(128) → Dense(13)
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ PHASE 1 (up to 15 epochs)                           │
│                                                     │
│  For each epoch:                                    │
│    For each batch (33 batches of 32 images):        │
│      1. Augment image (flip, rotate, etc.)          │
│      2. Forward pass → prediction                   │
│      3. Compute cross-entropy loss                  │
│      4. Backprop → gradients                        │
│      5. Adam updates ONLY Dense layers              │
│    ↓                                                │
│    Validate on 260 images → print accuracy          │
│    Save model if best val_accuracy seen             │
│    Reduce LR if stuck (patience=3)                  │
│    Stop if no improvement for 5 epochs              │
└─────────────────────────────────────────────────────┘
  │
  ▼
Unfreeze last 30 MobileNetV2 layers
Recompile with LR = 0.0001
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ PHASE 2 (up to 15 more epochs)                      │
│  Same loop, but now 30 more layers update too       │
└─────────────────────────────────────────────────────┘
  │
  ▼
Save celebrity_model.h5
Save celebrity_model_best.h5  (best epoch's weights)
Save model_info.json          (all metrics + class names)
  │
  ▼
Done → Run: streamlit run app.py
```

---

## 📌 Key Concepts Summary

| Concept | Plain English |
|---|---|
| **Pixel** | One tiny square of colour — just 3 numbers (R, G, B 0–255) |
| **Normalization** | Dividing pixels by 255 to get numbers between 0 and 1 |
| **Convolution** | A small filter sliding across the image, detecting patterns |
| **Padding** | Adding zeros around image edges so convolution doesn't shrink it |
| **Filter / Kernel** | A small learnable grid of numbers that detects one type of feature |
| **Feature Map** | The output of one filter applied to an image — shows where that feature appears |
| **ReLU** | "If negative, make zero" — adds non-linearity to enable complex learning |
| **Dense Layer** | Every output connects to every input — combines all features |
| **Softmax** | Converts raw numbers into probabilities that sum to 1.0 |
| **Loss** | A number measuring how wrong the model is — we minimize this |
| **Gradient** | Direction and slope — tells us which way to adjust each weight |
| **Backpropagation** | Using calculus to compute gradients for every weight |
| **Optimizer (Adam)** | Algorithm that updates weights using gradients |
| **Learning Rate** | How big a step to take when adjusting weights |
| **Epoch** | One complete pass through all training images |
| **Batch** | A small group of images processed together |
| **Augmentation** | Randomly transforming training images to create variety |
| **Overfitting** | Model memorizes training images but fails on new ones |
| **Dropout** | Randomly ignoring neurons during training to prevent overfitting |
| **Transfer Learning** | Reusing a model trained on one task (ImageNet) for another (celebrities) |
| **Fine-tuning** | Slowly adjusting pre-trained weights for the new task |
| **Weights** | The learnable numbers inside every layer — what training changes |
| **Callback** | A function that automatically runs at training milestones |
| **One-hot Encoding** | Representing a class as a vector of zeros with one 1 |
| **Cross-entropy Loss** | Penalizes wrong confident predictions heavily |
| **Val Accuracy** | Accuracy on images the model NEVER saw during training |
