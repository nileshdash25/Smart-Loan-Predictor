import tensorflow as tf

print("Loading heavy model...")
model = tf.keras.models.load_model('predictor/ml_files/loan_model.keras')

print("Converting to Lite version...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Ekdum sahi folder mein file save kar rahe hain
with open('predictor/ml_files/loan_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model perfectly saved!")