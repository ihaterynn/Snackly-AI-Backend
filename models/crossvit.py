import tensorflow as tf
from tensorflow.keras import layers

class CrossViT(tf.keras.Model):
    def __init__(self, num_classes=2):
        super(CrossViT, self).__init__()

        # Two branches: one with small patches, one with large patches
        self.small_patch_branch = self._make_vit_branch(patch_size=7)
        self.large_patch_branch = self._make_vit_branch(patch_size=14)
        
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        small_branch_output = self.small_patch_branch(inputs)
        large_branch_output = self.large_patch_branch(inputs)
        
        # Concatenate the outputs from both branches
        x = tf.concat([small_branch_output, large_branch_output], axis=-1)
        return self.fc(x)

    def _make_vit_branch(self, patch_size):
        return tf.keras.Sequential([
            layers.Conv2D(128, (patch_size, patch_size), padding='same'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu')
        ])

# Usage
if __name__ == "__main__":
    model = CrossViT(num_classes=2)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
