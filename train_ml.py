import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# -------------------------------
# Sample dataset (replace with your full dataset)
# -------------------------------
informal_sentences = [
    "तु काय करतोस?",
    "काय चाललंय?",
    "चल निघूया."
]
formal_sentences = [
    "तुम्ही काय करता?",
    "काय चाललंय का?",
    "चल निघूया."
]

# -------------------------------
# Tokenization
# -------------------------------
tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(informal_sentences + formal_sentences)
vocab_size = len(tokenizer.word_index) + 1

# Convert sentences to sequences
X_seq = tokenizer.texts_to_sequences(informal_sentences)
y_seq = tokenizer.texts_to_sequences(formal_sentences)

max_len = max(max(len(seq) for seq in X_seq), max(len(seq) for seq in y_seq))

X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post')
y_pad = pad_sequences(y_seq, maxlen=max_len, padding='post')

# -------------------------------
# Build simple sequence-to-sequence model
# -------------------------------
input_tensor = Input(shape=(max_len,))
embedding = Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True)(input_tensor)
lstm_out = LSTM(64, return_sequences=True)(embedding)
output = Dense(vocab_size, activation='softmax')(lstm_out)

model = Model(inputs=input_tensor, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# Train
# -------------------------------
model.fit(X_pad, np.expand_dims(y_pad, -1), epochs=20, batch_size=2)

# -------------------------------
# ✅ Save model in Keras 3 format
# -------------------------------
model.save("marathi_formalizer.keras")   # <- final model file

print("✅ Model trained and saved successfully as 'marathi_formalizer.keras'")

# -------------------------------
# Save tokenizer separately
# -------------------------------
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer saved successfully!")
