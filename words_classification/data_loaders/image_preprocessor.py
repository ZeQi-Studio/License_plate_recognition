import os
import logging
import cv2
import numpy as np
import random

from templates import PreprocessorTemplate, ConfigTemplate

logger = logging.getLogger(__name__)


class ImagePreprocessor(PreprocessorTemplate):
    def __init__(self, config):
        super(ImagePreprocessor, self).__init__(config)
        self.image_dict = {}

    def preprocess(self):
        self.__load_init_set()
        self.__image_augmentation()

    def __load_init_set(self):
        self.config: ImagePreprocessorConfig

        for char in self.config.CHARACTER_LIST:
            image_file = os.path.join(self.config.DATASET_FILE_ROOT, str(char) + ".png")
            logger.debug("Loading image from file: %s", image_file)

            image = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), flags=0)

            if image is None:
                logger.warning("Try to read image %s failed! Skip.", image_file)

            self.image_dict[str(char)] = image

    def __image_augmentation(self):
        self.config: ImagePreprocessorConfig
        random.seed(1000)
        for char, image in self.image_dict.items():
            for _ in range(self.config.AUGMENTATION_AMOUNT):
                image = self.__brightness(image, random.random() * 0.4 + 0.8)

                image = cv2.resize(image, self.config.OUT_IMAGE_SIZE)

                cv2.imshow("after aug", image)
                cv2.waitKey(1)

    @staticmethod
    def __brightness(image, weight: float):
        dst = np.array(image.copy(), dtype=np.float)

        dst = (dst * weight)
        dst = dst.clip(0, 255)

        return dst.astype(np.uint8)


class ImagePreprocessorConfig(ConfigTemplate):
    def __init__(self,
                 dataset_file_root,
                 character_list,
                 augmentation_amount,
                 out_image_size
                 ):
        self.DATASET_FILE_ROOT = dataset_file_root
        self.CHARACTER_LIST = character_list
        self.AUGMENTATION_AMOUNT = augmentation_amount
        self.OUT_IMAGE_SIZE = out_image_size


# only test case
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    my_conf = ImagePreprocessorConfig(
        dataset_file_root="dataset/character_data/",
        character_list=[str(i) for i in range(10)] + list("QWERTYUPASDFGHJKLZXCVBNM") + [
            '冀', '新', '鄂', '宁', '桂', '黑', '湘', '皖', '云', '豫', '蒙', '赣', '吉', '辽', '苏', '甘', '晋', '浙', '闽',
            '渝', '贵', '陕', '粤', '川', '鲁', '琼', '青', '藏', '京', '津', '沪'],
        augmentation_amount=100,
        out_image_size=(13 * 5, 25 * 5)
    )

    my_preprocessor = ImagePreprocessor(my_conf)

    my_preprocessor.preprocess()
