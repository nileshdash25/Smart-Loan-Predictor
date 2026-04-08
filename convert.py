import tensorflow as tf

# Apna purana model load karo
model = tf.keras.models.load_model('loan_website/predictor/ml_files/loan_model.keras')

# TFLite mein convert karo
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Naya halka model save karo
with open('loan_website/predictor/ml_files/loan_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Jhakaas! Model Lite version mein convert ho gaya!")