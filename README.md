# 🧠 Handwritten Digit Classifier Sandbox

A mobile-friendly, high-contrast interactive sandbox web application built with Streamlit that handles real-time digit recognition. The app runs a drawing canvas input pipeline against three distinct, custom-trained architectures: a Single-Layer Perceptron, a Multi-Layer Artificial Neural Network (ANN), and a Deep Convolutional Neural Network (CNN).

---

## 🚀 Features

*   **Interactive Drawing Surface:** High-contrast `280x280` canvas sandbox built to emulate the geometric distribution properties of the MNIST dataset.
*   **Triple-Model Execution Pipeline:** Feeds your drawing concurrently to three isolated architectures for side-by-side performance comparisons.
*   **Accessible Modern UI:** Custom glassmorphism layout tailored for both mobile layouts and dark mode environments.
*   **Visual Debugger Engine:** Includes a downsampling expander component displaying exactly how the internal neural layer arrays reshape and normalize your live input.

---

## 🧱 System Architecture & Pipelines

The application processes your drawing through a standardized pipeline before feeding it to three model architectures:

```
User Drawing (280×280)
         ↓
   Resize to 28×28
         ↓
   Normalize [0,1]
         ↓
    ┌──────────┬────┬
    ↓          ↓    ↓
   🔵         🟣   🟡
  Perceptron  ANN  CNN
  
```

**Pipeline Flow:**
1. **Raw Canvas:** Capture freehand digit (280×280 pixels)
2. **Resize:** Downsample to MNIST standard (28×28)
3. **Normalize:** Scale pixel values from [0, 255] to [0.0, 1.0]
4. **Predict:** Feed normalized input to all three models simultaneously


### Model Performance Profiles

| Architecture Model | Targeted Data Input Shape | Structural Description | Best Suite Capabilities |
| :--- | :--- | :--- | :--- |
| **Perceptron** | `(1, 28, 28)` | Baseline Single Dense Layer + Softmax output map. | Linear classifications. Highly vulnerable to drawing shifts or off-center strokes. |
| **ANN (Deep)** | `(1, 28, 28)` | Multi-stage fully connected Dense stack with ReLU drops. | Non-linear boundaries. High convergence speed with standard variations. |
| **CNN (Optimal)** | `(1, 28, 28, 1)` | Convolutional filtering stages + MaxPool dimensional pooling layers. | **Highly Robust.** Tracks spatial relationships and contours. Resilient to drawing off-center. |

---

## 🛠️ Setup & Installation

The app requires **Python 3.10** and TensorFlow with Keras support. I recommend using the Conda environment for reliable compatibility.


### 1. Install Dependencies
```bash
cd /path/to/MNIST-Digit-Classifier
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

The app will launch on `http://localhost:8501` by default. If that port is in use, specify an alternative:

```bash
streamlit run app.py --server.port 8503
```

---

## 📁 Project Structure

```
MNIST-Digit-Classifier/
├── app.py                  # Streamlit UI, canvas, model loading & predictions
├── requirements.txt        # Dependencies (TensorFlow, Streamlit, NumPy, Pillow)
├── README.md               # This is documentation
└── models/                 # Trained model weights
    ├── perceptron_model.h5 # Single-layer baseline
    ├── ann_model.h5        # Multi-layer deep ANN
    └── cnn_model.h5        # Convolutional neural network (best accuracy)
```

---

