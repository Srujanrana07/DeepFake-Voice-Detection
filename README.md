# 🔊 DeepFake Voice Detection  
A robust machine learning system for identifying **AI-generated (deepfake) voices** using MFCC features, spectral audio statistics, and a combination of **Deep Learning (CNN)** and **Classical ML (Random Forest, KNN)** models.

---

## ⭐ Key Features  
- Detects **real vs fake** speech with **98%+ accuracy**  
- Uses **MFCCs**, **Mel-spectrogram statistics**, and **spectral features**  
- Implements **1D CNN**, **Random Forest**, and **KNN**  
- Includes full evaluation metrics with placeholders for all plots  
- Built to support future deployment for fraud and security applications  

---

## 📂 Dataset  
- Audio format: **16 kHz, mono WAV**  
- Classes: **Real**, **Fake**  
- Includes various **SNR levels** and **noise-reduction methods**  
- Dataset contains **class imbalance**, handled using **SMOTE**  

---

## 🎧 Feature Extraction  

### **MFCC Features (40-D)**
- Extracted using Librosa  
- Mean-pooled across time  
- Used as input to the 1D CNN  

### **Engineered Audio Features (318-D)**
- MFCC means  
- Mel-Spectrogram means  
- Log-Spectrogram (STFT) means  
- Used for Random Forest and KNN models  

---

## 🏗️ System Architecture  

```
           ┌─────────────── Preprocessing ───────────────┐
           │                                               │
Audio → Load → Normalize → MFCC / Spectrogram Extraction → Features
           │                                               │
           └──────────────────────┬────────────────────────┘
                                  │
       ┌──────────────────────────┴──────────────────────────┐
       │                                                     │
  318-D Engineered Features                          MFCC Map (40×1)
       │                                                     │
Random Forest / KNN                                     1D CNN Model
       │                                                     │
       └──────────────────────────┬──────────────────────────┘
                                  ↓
                           **Real / Fake**
```

---

## 🧠 Models Implemented  

# **1️⃣ 1D CNN (MFCC-Based)**  

### ✔ Architecture  
- Conv1D → Dropout  
- MaxPooling  
- Conv1D → Dropout  
- Dense + Softmax  
- Trained for 40 epochs  

### ✔ Features Used
`![CNN Features](/assets/mfcc-40.png)`

### ✔ Performance  
- **Dev Accuracy:** ~86%  
- **Eval Accuracy:** ~88%  
- Strong on detecting **fake audio**  
- Mild overfitting  

---

## 📊 CNN Evaluation (Placeholders)

### 🟦 Training Accuracy Curve  
`![CNN Accuracy Curve](/assets/accuracy.png)`

### 🟥 Training Loss Curve  
`![CNN Loss Curve](/assets/loss.png)`

### 🟩 Confusion Matrix (CNN)  
`![CNN Confusion Matrix](/assets/confussionmatrix-cnn.png)`

### 🟪 ROC Curve (CNN)  
`![CNN ROC Curve](/assets/roc.png)`

### 🟨 Precision–Recall Curve (CNN)  
`![CNN PR Curve](/assets/presision and recall.png)`

---
# ✔ Features Used For the 318 Dimention
`![RN & KNN Features](/assets/realvsfake.png)`

# **2️⃣ Random Forest Classifier**

### ✔ Performance  
- **Accuracy:** **98.82%**  
- High precision & recall on both classes  
- Extremely robust to noise & dataset variance  



### 🟩 Confusion Matrix (RF)  
`![RF Confusion Matrix](/assets/confusion.png)`

---

# **3️⃣ K-Nearest Neighbours (KNN)**  

### ✔ Performance  
- **Accuracy:** **98.29%**  
- Very stable across different samples  
- k = 7 chosen for optimal performance  

### 🟩 Confusion Matrix (KNN)  
`![KNN Confusion Matrix](/assets/download.png)`

---

## 📊 Overall Model Comparison  

| Model | Accuracy | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| **Random Forest** | ⭐ **98.82%** | Best overall, robust to noise | Slow to train on huge datasets |
| **KNN (k=7)** | 98.29% | Simple & competitive | Slow inference on large data |
| **CNN (MFCCs)** | ~88% | Learns temporal patterns | Overfitting risk |

---

## 🚧 Limitations  
- Dataset imbalance required oversampling  
- CNN performance limited by MFCC-only representation  
- Needs evaluation on unseen deepfake generators  
- Real-world recordings with background noise not fully tested  

---

## 🚀 Future Enhancements  
- Use **2D CNNs** on spectrogram images  
- Add transformer-based encoders (**wav2vec 2.0**, **HuBERT**, **Whisper**)  
- Deploy as a **web or mobile app** for live detection  
- Add adversarial robustness  
- Add explainable AI for forensic usage  

---

## 🛠️ Tech Stack  
- **Python**  
- **Librosa** – audio processing  
- **TensorFlow / Keras** – CNN model  
- **scikit-learn** – RF, KNN, SMOTE  
- **NumPy / pandas** – preprocessing  

---

## 🙌 Contributors  
**Srujan Rana**  

---
