import tensorflow as tf

class PatchMerger(tf.keras.layers.Layer):
    def __init__(self):
        super(PatchMerger, self).__init__()

    def call(self, small_patch_output, large_patch_output):
        # Merge small and large patch outputs
        merged_patches = tf.concat([small_patch_output, large_patch_output], axis=-1)
        return merged_patches

# Usage in CrossViT
if __name__ == "__main__":
    patch_merger = PatchMerger()
    small_patch = tf.random.normal([32, 128])
    large_patch = tf.random.normal([32, 128])
    
    merged_output = patch_merger(small_patch, large_patch)
    print(merged_output.shape)
