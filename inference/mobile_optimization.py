import tensorflow as tf

def convert_to_tflite(model_path, output_path):
    # Load trained model
    model = tf.keras.models.load_model(model_path)
    
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the converted model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved at: {output_path}")

# Example usage
if __name__ == "__main__":
    convert_to_tflite('saved_models/hybrid_model.h5', 'saved_models/hybrid_model.tflite')
