import tensorflow as tf

def load_mobilenetv2(input_shape=(224, 224, 3)):
    """Load MobileNetV2 as a feature extractor."""
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model
    return base_model
