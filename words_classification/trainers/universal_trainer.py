import tensorflow as tf

from templates.trainer_template import TrainerTemplate


class UniversalTrainer(TrainerTemplate):
    def __init__(self, model, data, config):
        super(UniversalTrainer, self).__init__(model, data, config)

    def train(self, *args):
        self.model: tf.keras.Model
        self.data: tf.data.Dataset
        self.config: UniversalTrainerConfig

        tf_board_callbacks = tf.keras.callbacks.TensorBoard(log_dir="log/" + self.timestamp,
                                                            histogram_freq=100,
                                                            update_freq='epoch')

        self.model.fit(x=self.data,
                       epochs=self.config.EPOCH,
                       callbacks=[tf_board_callbacks])

    def evaluate(self, eval_set):
        self.model: tf.keras.Model

        return self.model.evaluate(eval_set)


class UniversalTrainerConfig:
    def __init__(self,
                 epoch):
        self.EPOCH = epoch
