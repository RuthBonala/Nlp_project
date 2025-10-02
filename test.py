# test.py
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load model (Keras 3 format)
# -------------------------------
model = tf.keras.models.load_model("marathi_formalizer.keras", compile=False)

# -------------------------------
# Load tokenizer
# -------------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Reverse word index for decoding
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

# -------------------------------
# Function to formalize sentence
# -------------------------------
def formalize_sentence(sentence):
    # Convert input to sequence
    seq = tokenizer.texts_to_sequences([sentence])
    seq_padded = pad_sequences(seq, maxlen=model.input_shape[1], padding='post')
    
    # Predict
    pred = model.predict(seq_padded)

    # Handle sequence output (batch_size, seq_len, vocab_size)
    pred_indices = np.argmax(pred[0], axis=-1)
    formal_sentence = " ".join(
        [reverse_word_index.get(idx, "") for idx in pred_indices if idx != 0]
    )

    return formal_sentence.strip()

# -------------------------------
# Interactive loop
# -------------------------------
print("✅ Marathi Formalizer Ready! Type 'exit' to quit.")
while True:
    sentence = input("Enter informal sentence: ").strip()
    if sentence.lower() == "exit":
        break
    formal_sentence = formalize_sentence(sentence)
    print(f"Formal: {formal_sentence}\n")
