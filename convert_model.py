import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

# Define your custom layer exactly as in training
class NotEqual(tf.keras.layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.not_equal(x, y)

# Load the .h5 model using custom_object_scope
with custom_object_scope({'NotEqual': NotEqual}):
    model = load_model("marathi_formalizer_model.h5", compile=False)

# Save in the new SavedModel format
model.save("marathi_formalizer_model_saved")  # folder, not .h5
print("✅ Model converted and saved in SavedModel format!")
