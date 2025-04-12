import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import string
import os

# --------------------------------
# 1. Load and Preprocess Text
# --------------------------------

def load_text(file_path):
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            return f.read()
    except Exception:
        with open(file_path, "r", encoding='latin1') as f:
            return f.read()

text = load_text("test.txt").lower()
text = text.translate(str.maketrans('', '', string.punctuation)).replace('\n', ' ')
print(f"Loaded text length: {len(text)}")

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
print(f"Total unique words: {total_words}")

# Create input sequences
sequence_length = 10
words = text.split()
input_sequences = []

for i in range(len(words) - sequence_length):
    seq = words[i:i + sequence_length + 1]
    input_sequences.append(seq)

# Convert to numerical sequences
sequences = tokenizer.texts_to_sequences([' '.join(seq) for seq in input_sequences])
sequences = np.array(sequences)

X, y = sequences[:, :-1], sequences[:, -1]  # No one-hot encoding

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print(f"Training sequences: {X_train.shape}, Validation sequences: {X_val.shape}")

# --------------------------------
# 2. Build the Model
# --------------------------------

def create_model(vocab_size, input_len, embedding_dim=100, lstm_units=150):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_len),
        LSTM(lstm_units, return_sequences=True),
        Dropout(0.2),
        LSTM(lstm_units),
        Dropout(0.2),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',  # Use sparse loss
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

model = create_model(total_words, X.shape[1])
model.summary()

# --------------------------------
# 3. Train the Model
# --------------------------------

os.makedirs("model_checkpoints", exist_ok=True)

checkpoint = ModelCheckpoint("model_checkpoints/best_model.h5", monitor='val_accuracy',
                             save_best_only=True, verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=128,
    callbacks=[checkpoint, early_stopping]
)

# --------------------------------
# 4. Plot Training Results
# --------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# --------------------------------
# 5. Text Generation
# --------------------------------

def generate_text(model, tokenizer, seq_length, seed_text, next_words=20, temperature=1.0):
    output = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output])[0][-seq_length:]
        token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')
        predictions = model.predict(token_list, verbose=0)[0]
        predictions = np.log(predictions + 1e-8) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))
        predicted_id = np.random.choice(len(predictions), p=predictions)
        for word, index in tokenizer.word_index.items():
            if index == predicted_id:
                output += ' ' + word
                break
    return output

# --------------------------------
# 6. Generate Samples
# --------------------------------

seeds = [
    "to be or not to",
    "all the world is a",
    "what light through yonder"
]

for temp in [0.7, 1.0]:
    print(f"\n=== Temperature: {temp} ===\n")
    for seed in seeds:
        print(f"Seed: {seed}")
        print("Generated:", generate_text(model, tokenizer, sequence_length, seed, 20, temperature=temp))
        print("-" * 50)

print("LSTM Text Generation Completed.")
