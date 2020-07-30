import tensorflow as tf
from templates import ModelTemplate, ConfigTemplate


class CNNModel(ModelTemplate):

    def build(self, *args):
        self.config: CNNModelConfig
        image_input = tf.keras.Input(shape=self.config.INPUT_SHAPE)

        hidden_layer = tf.keras.layers.Dropout(rate=0.5)(image_input)
        hidden_layer = tf.keras.layers.Conv2D(filters=16,
                                              kernel_size=(2, 2),
                                              strides=(1, 1),
                                              padding="SAME",
                                              activation="relu")(hidden_layer)
        hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=32,
                                              kernel_size=(2, 2),
                                              strides=(1, 1),
                                              padding="SAME",
                                              activation="relu")(hidden_layer)
        hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=(8, 8),
                                              strides=(4, 4),
                                              padding="SAME",
                                              activation="relu")(hidden_layer)
        hidden_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                 padding="SAME")(hidden_layer)
        hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
        hidden_layer = tf.keras.layers.Flatten()(hidden_layer)

        hidden_layer = tf.keras.layers.Dense(128,
                                             activation="relu")(hidden_layer)
        hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
        hidden_layer = tf.keras.layers.Dense(self.config.OUTPUT_LEN,
                                             activation="relu")(hidden_layer)
        output = tf.keras.layers.Softmax()(hidden_layer)

        self.model = tf.keras.Model(inputs=image_input, outputs=output)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE, clipnorm=0.5),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                                    tf.keras.metrics.SparseCategoricalCrossentropy()])


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

    my_model.show_summary()
