# EEG Eye State Classification using LSTM

A Deep Learning time-series classification pipeline designed to predict eye state (Open vs. Closed) from continuous 14-channel EEG (Electroencephalography) signals using Long Short-Term Memory (LSTM) networks.

---

## 📌 Project Overview

This project implements an end-to-end Recurrent Neural Network (RNN/LSTM) pipeline to analyze temporal EEG signals and accurately classify whether a subject's eyes are open or closed. The pipeline addresses common time-series challenges such as temporal dependencies, signal noise, and data distribution shifts.

---

## 📊 Dataset Information

- **Source:** [UCI Machine Learning Repository - EEG Eye State Dataset](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)
- **Signal Attributes:** 14 continuous EEG electrode measurements (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4).
- **Target Label (`eyeDetection`):**
  - `0`: Eyes Open
  - `1`: Eyes Closed
- **Temporal Nature:** Continuous measurement over 117 seconds (~14,980 samples).

---

## 🛠️ Pipeline Architecture

1. **Exploratory Data Analysis (EDA):** Signal inspection, class distribution verification, and descriptive statistics.
2. **Data Preprocessing & Cleaning:**
   - Outlier detection and handling using the Interquartile Range (IQR) technique.
   - Feature normalization with `StandardScaler` to ensure stable gradient propagation.
3. **Temporal Sequence Engineering:**
   - Transformed continuous tabular signals into fixed-length sequential windows (`timesteps = 10`) for recurrent modeling.
4. **Model Architecture (Deep LSTM):**
   - **Input Layer:** Multi-feature sequential input `(batch_size, timesteps, features)`.
   - **LSTM Layers:** Captures short and long-term temporal dependencies.
   - **Regularization:** Integrated `BatchNormalization` and `Dropout` (0.2–0.3) to mitigate overfitting.
   - **Dense Output Layer:** Single neuron with Sigmoid activation for binary classification.
5. **Training & Optimization:**
   - Loss Function: `BinaryCrossentropy`
   - Optimizer: `Adam` (with learning rate scheduling / EarlyStopping)
   - Validation Strategy: Stratified splitting to maintain balanced class distributions.

---

## 📈 Results & Performance

- **Test Accuracy:** ~98.4%
- **Validation Loss:** ~0.078
- **Evaluation Metrics:** High Precision, Recall, and F1-score across both Open and Closed states without temporal data leakage.

---

## 🧰 Tech Stack

- **Language:** Python 3.10+
- **Deep Learning:** TensorFlow / Keras (`Sequential`, `LSTM`, `BatchNormalization`, `Dropout`, `Dense`)
- **Data Manipulation & Preprocessing:** Pandas, NumPy, Scikit-Learn (`StandardScaler`, `train_test_split`)
- **Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure
```text
├── EEG-Eye-State.csv                  # Dataset file
├── EEG_Eye_State_Classification.ipynb # Complete end-to-end Jupyter Notebook
├── models/
│   └── eeg_lstm_model.keras          # Serialized trained model
├── README.md                          # Project documentation
└── requirements.txt                   # Dependency list
