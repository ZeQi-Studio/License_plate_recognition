import tensorflow as tf
import logging
import numpy as np
from templates import DataLoaderTemplate, ConfigTemplate

logger = logging.getLogger(__name__)


class ImageSetDataLoader(DataLoaderTemplate):
    def __init__(self, config):
        self.image_dict = None
        self.label_dict = None
        super(ImageSetDataLoader, self).__init__(config)

    def load(self, *args):
        self.config: ImageSetDataLoaderConfig
        self.image_dict = self.config.IMAGE_DICT
        self.label_dict = {k: index for index, (k, v) in enumerate(self.image_dict.items())}

        logger.info("Label dict: %s", self.label_dict)

        self.dataset = tf.data.Dataset.from_generator(
            self.__data_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(self.config.IMAGE_SHAPE, (len(self.label_dict.items())))
        ).shuffle(self.config.SHUFFLE_BUFFER_SIZE).batch(self.config.BATCH_SIZE)
        self.dataset.prefetch(10)

    def __data_generator(self):
        for char, image_list in self.image_dict.items():
            for image in image_list:
                yield np.reshape(image / 255, self.config.IMAGE_SHAPE), \
                      np.eye(len(self.label_dict.items()), dtype=np.float)[self.label_dict[char]]


class ImageSetDataLoaderConfig(ConfigTemplate):
    def __init__(self,
                 image_dict,
                 image_shape,
                 batch_size,
                 shuffle_buffer_size
                 ):
        self.IMAGE_DICT = image_dict
        self.IMAGE_SHAPE = image_shape
        self.BATCH_SIZE = batch_size
        self.SHUFFLE_BUFFER_SIZE = shuffle_buffer_size
        assert batch_size >= 1


if __name__ == '__main__':
    import cv2

    CHARACTER_LIST = [str(i) for i in range(10)] + list("QWERTYUPASDFGHJKLZXCVBNM") + [
        '冀', '新', '鄂', '宁', '桂', '黑', '湘', '皖', '云', '豫', '蒙', '赣', '吉', '辽', '苏', '甘', '晋', '浙', '闽',
        '渝', '贵', '陕', '粤', '川', '鲁', '琼', '青', '藏', '京', '津', '沪']

    my_config = ImageSetDataLoaderConfig("dataset/character_data_augmentation/dataset_dict_dump.pickle",
                                         (125, 65, 1),
                                         1, 10000, )
    my_dataset = ImageSetDataLoader(my_config)

    for my_image, my_label in my_dataset.get_dataset():
        print((my_image[0].numpy()))
        print(np.argmax(my_label, axis=-1)[0])
        print(CHARACTER_LIST[np.argmax(my_label, axis=-1)[0]])
        cv2.imshow("image", my_image[0].numpy())
        cv2.waitKey(1000)
