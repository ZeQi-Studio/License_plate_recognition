import tensorflow as tf

from templates import DataLoaderTemplate, ConfigTemplate


class ImageSetDataLoader(DataLoaderTemplate):
    def __init__(self, config):
        self.image_dict = None
        self.label_dict = None
        super(ImageSetDataLoader, self).__init__(config)

    def load(self, *args):
        self.config: ImageSetDataLoaderConfig
        self.image_dict = self.cache_load(self.config.PICKLE_DUMPED_FILE)
        self.label_dict = {k: [index, ] for index, (k, v) in enumerate(self.image_dict.items())}

        self.dataset = tf.data.Dataset.from_generator(self.__data_generator,
                                                      output_types=(tf.float32, tf.int8),
                                                      output_shapes=(self.config.IMAGE_SHAPE, (1,)))

    def __data_generator(self):
        for char, image_list in self.image_dict.items():
            for image in image_list:
                yield image, self.label_dict[char]


class ImageSetDataLoaderConfig(ConfigTemplate):
    def __init__(self,
                 pickle_dumped_file,
                 image_shape
                 ):
        self.PICKLE_DUMPED_FILE = pickle_dumped_file
        self.IMAGE_SHAPE = image_shape


if __name__ == '__main__':
    my_config = ImageSetDataLoaderConfig("dataset/character_data_augmentation/dataset_dict_dump.pickle",
                                         (125, 65))
    my_dataset = ImageSetDataLoader(my_config)

    for image, label in my_dataset.get_dataset():
        print(image, label)
