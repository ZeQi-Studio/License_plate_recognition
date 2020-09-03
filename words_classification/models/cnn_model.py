import tensorflow as tf
from templates import ModelTemplate, ConfigTemplate


class CNNModel(ModelTemplate):

    def build(self, *args):
        self.config: CNNModelConfig
        image_input = tf.keras.Input(shape=self.config.INPUT_SHAPE)

        hidden_layer = tf.keras.layers.Dropout(rate=0.1)(image_input)
        hidden_layer = tf.keras.layers.Conv2D(filters=32,
                                              kernel_size=(4, 4),
                                              strides=(1, 1),
                                              padding="SAME",
                                              activation="relu")(hidden_layer)
        hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=(8, 8),
                                              strides=(2, 2),
                                              padding="SAME",
                                              activation="relu")(hidden_layer)
        hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=128,
                                              kernel_size=(4, 4),
                                              strides=(2, 2),
                                              padding="SAME",
                                              activation="relu")(hidden_layer)
        hidden_layer = tf.keras.layers.MaxPool2D(pool_size=(4, 4),
                                                 padding="SAME")(hidden_layer)
        hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
        hidden_layer = tf.keras.layers.Flatten()(hidden_layer)

        hidden_layer = tf.keras.layers.Dense(1024,
                                             activation="relu")(hidden_layer)
        hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
        hidden_layer = tf.keras.layers.Dense(self.config.OUTPUT_LEN,
                                             activation="relu")(hidden_layer)
        output = tf.keras.layers.Softmax()(hidden_layer)

        self.model = tf.keras.Model(inputs=image_input, outputs=output)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE, clipnorm=1),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=[tf.keras.metrics.CategoricalAccuracy(),
                                    tf.keras.metrics.CategoricalCrossentropy()])

        self.show_summary()


class CNNModelConfig(ConfigTemplate):
    def __init__(self,
                 input_shape,
                 output_len,
                 learning_rate):
        self.INPUT_SHAPE = input_shape
        self.OUTPUT_LEN = output_len
        self.LEARNING_RATE = learning_rate


if __name__ == '__main__':
    my_config = CNNModelConfig((125, 65, 3),
                               10,
                               0.001)
    my_model = CNNModel(my_config)
