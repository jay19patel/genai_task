# 🧠 LSTM Text Generator (Word-Level)

This project trains a word-level LSTM (Long Short-Term Memory) neural network on a text corpus (like Shakespeare) to generate text in a similar style.

---

## 🚀 Features

- Word-level tokenization using Keras Tokenizer
- LSTM neural network with two stacked LSTM layers
- Text generation with temperature-based sampling
- Training loss/accuracy visualization
- Save & resume best model checkpoints

---

## 📁 Folder Structure

```
project/
│
├── test.txt               # Input text corpus (e.g., Shakespeare works)
├── app.py                 # Main training & generation script
├── model_checkpoints/     # Folder for saving the best model
└── README.md              # This file
```

---

## 🔧 Requirements

Python 3.7 or later  
Install required libraries:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, you can install manually:

```bash
pip install numpy tensorflow matplotlib scikit-learn
```

---

## 📦 How to Use

### 1. Add Your Dataset
Replace `test.txt` with your own `.txt` file (must be plain text).

### 2. Train the Model

```bash
python app.py
```

This will:
- Train the LSTM model
- Save the best model in `model_checkpoints/best_model.h5`
- Plot accuracy and loss
- Generate sample text after training

### 3. Generate Text (After Training)

After the model is trained, the script automatically generates text using sample seed phrases like:

```text
Seed: to be or not to
Generated: to be or not to speak of things that make the world weep...
```

---

## 🔥 Tips

- You can adjust the `sequence_length`, `embedding_dim`, or `lstm_units` in `app.py` for better results.
- The `temperature` setting in generation controls randomness:
  - Lower (e.g., 0.7) = more predictable
  - Higher (e.g., 1.2) = more creative/random

---

## 📚 Example Dataset

You can use:
- Shakespeare: https://www.gutenberg.org/ebooks/100
- Song lyrics
- Any plain text file (make sure it's large enough)

---

## 🧠 Output Sample

```
Seed: what light through yonder
Generated: what light through yonder soul of mine shall shine the heavens apart
```

---

