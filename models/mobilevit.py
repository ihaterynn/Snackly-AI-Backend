import tensorflow as tf
from tensorflow.keras import layers

class MobileViT(tf.keras.Model):
    def __init__(self, num_classes=2):
        super(MobileViT, self).__init__()

        # Adjusting MobileNet block to work with smaller feature maps (7x7)
        self.mobile_block = self._make_mobilenet_block(input_shape=(7, 7, 1280))  # Use the shape coming from MobileNetV2
        
        # Vision Transformer block with LayerNormalization and Dropout
        self.vit_block = self._make_vit_block()

        # Add Dropout before Flattening
        self.dropout = tf.keras.layers.Dropout(0.4)

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.mobile_block(inputs)  # Use the feature maps directly, no need for reshape
        x = self.vit_block(x)
        x = self.dropout(x)
        x = self.flatten(x)
        return self.fc(x)

    def _make_mobilenet_block(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)  # Adjust the input shape based on MobileNetV2 output
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(inputs)  # Adjust filter sizes as needed
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return tf.keras.Model(inputs, x, name="mobile_block")

    def _make_vit_block(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, (1, 1), padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.4)
        ])
