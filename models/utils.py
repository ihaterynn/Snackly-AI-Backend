import tensorflow as tf

# Custom Layer Example
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CustomLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Example usage
if __name__ == "__main__":
    layer = CustomLayer(10)
    x = tf.random.normal([32, 10])
    output = layer(x)
    print(output.shape)
