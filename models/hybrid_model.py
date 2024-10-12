import tensorflow as tf
from .mobilevit import MobileViT  # Ensure correct relative import

class HybridModel(tf.keras.Model):
    def __init__(self, num_classes=2):
        super(HybridModel, self).__init__()

        # Use only MobileViT (simplifying by removing CrossViT)
        self.mobile_vit = MobileViT(num_classes=num_classes)

        # Final dense layer for binary classification
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')  # Use sigmoid for binary classification

    def call(self, inputs):
        # Get outputs from MobileViT
        mobile_output = self.mobile_vit(inputs)

        # Final classification layer
        return self.dense(mobile_output)

# Usage
if __name__ == "__main__":
    model = HybridModel(num_classes=2)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
