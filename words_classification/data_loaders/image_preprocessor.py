import os
import logging
import cv2
import numpy as np
import random

from tqdm import tqdm

from templates import PreprocessorTemplate, ConfigTemplate
from templates.utils import mkdir

logger = logging.getLogger(__name__)


class ImagePreprocessor(PreprocessorTemplate):
    def __init__(self, config):
        super(ImagePreprocessor, self).__init__(config)
        self.config: ImagePreprocessorConfig
        self.image_dict = {}

        self.preprocess()
        if self.config.SAVE_IMAGE_FORMAT_RESULT:
            self.save_result()
            self.cache_save(self.image_dict, os.path.join(self.config.OUT_FILE_ROOT, "dataset_dict_dump.pickle"))

    def preprocess(self):
        self.__load_init_set()
        self.__image_augmentation()

    def __load_init_set(self):
        self.config: ImagePreprocessorConfig

        for char in self.config.CHARACTER_LIST:
            image_file = os.path.join(self.config.DATASET_FILE_ROOT, str(char) + ".png")
            logger.debug("Loading image from file: %s", image_file)

            image = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), flags=cv2.IMREAD_COLOR)

            if image is None:
                logger.warning("Try to read image %s failed! Skip.", image_file)

            self.image_dict[str(char)] = image

    def __image_augmentation(self):
        self.config: ImagePreprocessorConfig

        logger.info("Generating augmentation data...")

        for char, org_image in tqdm(self.image_dict.items(), ncols=100):
            image_list = []
            for _ in range(self.config.AUGMENTATION_AMOUNT):
                image = org_image.copy()

                # cv2.imshow("before", image)
                # cv2.waitKey(1)

                image = self.__gray_scale(image)
                # image = self.__thresh_mean(image)
                image = cv2.GaussianBlur(image, (5, 5), random.random() * 50)
                image = self.__random_line(image)
                image = self.__rotate(image, 30)
                image = self.__random_line(image)
                image = self.__random_crop(image, 15)
                # image = self.__random_noise(image, random.randint(0, 20))
                # image = self.__contrast(image, random.random() * 0.8 + 0.9, random.randint(-50, 50))
                image = self.__thresh_half(image)
                image = cv2.resize(image, self.config.OUT_IMAGE_SIZE[:2])

                # cv2.imshow("after aug", image)
                # cv2.waitKey(100)

                image_list.append(image)
            self.image_dict[char] = image_list

    @staticmethod
    def __brightness(image, weight: float):
        dst = np.array(image.copy(), dtype=np.float)

        dst = (dst * weight)
        dst = dst.clip(0, 255)

        return dst.astype(np.uint8)

    @staticmethod
    def __rotate(image, angle):
        rows, cols = image.shape

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.randint(-angle, angle), random.random() * 0.3 + 0.4)

        dst = cv2.warpAffine(image, M, (cols, rows))
        return dst

    @staticmethod
    def __random_noise(image, strength):
        noise = np.random.normal(size=np.shape(image))
        with_noise = noise.astype(np.uint8) * (random.randint(-strength, strength)) + image

        return np.clip(with_noise, 0, 255).astype(np.uint8)

    @staticmethod
    def __random_crop(image, border: int):
        x1 = random.randint(0, border)
        x2 = random.randint(0, border)
        y1 = random.randint(0, border)
        y2 = random.randint(0, border)

        shape = np.shape(image)

        image = image[x1:shape[0] - x2, y1:shape[1] - y2, ...]
        # logger.debug("%s%s", shape, image)
        image = cv2.resize(image, shape[:2])

        return image

    @staticmethod
    def __gray_scale(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def __contrast(image, c, b):
        """
        给每个像素的每个通道增加亮度b
        :param image:输入图片
        :param c:
        :param b:
        :return:
        """
        shape = image.shape

        # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
        blank = np.zeros(shape, image.dtype)
        dst = cv2.addWeighted(image, c, blank, 1 - c, b)

        return dst

    @staticmethod
    def __thresh_mean(image):
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 0)

    @staticmethod
    def __thresh_half(image):
        thresh, ret = cv2.threshold(image, random.randint(100, 200), 255, cv2.THRESH_BINARY)
        return ret

    @staticmethod
    def __random_line(image):
        shape = np.shape(image)
        for i in range(5):
            x = random.randint(0, shape[0] - 2)
            y = random.randint(0, shape[1] - 2)
            image[x:x + random.randint(0, 2), y:y + random.randint(0, int(shape[1] / 2)), ...] = 255

        for i in range(2):
            x = random.randint(0, shape[0] - 2)
            y = random.randint(0, shape[1] - 2)
            image[x:x + random.randint(0, 3), y:y + random.randint(0, int(shape[1])), ...] = 255

        for i in range(2):
            x = random.randint(0, shape[0] / 2)
            y = random.randint(0, shape[1] / 2)
            image[x:x + random.randint(0, 3), 0:y + random.randint(30, int(shape[1])), ...] = 255

        for i in range(2):
            x = random.randint(0, shape[0] / 2)
            y = random.randint(0, shape[1] / 2)
            image[int(shape[0] / 2) + x:int(shape[0] / 2) + x + random.randint(1, 3),
            0:y + random.randint(30, int(shape[1])), ...] = 255

        return image

    def save_result(self):
        self.config: ImagePreprocessorConfig
        logger.info("Saving augmentation image to disk...")

        for char, image_list in tqdm(self.image_dict.items(), ncols=100):
            for index, image in enumerate(image_list):
                save_path = os.path.join(self.config.OUT_FILE_ROOT, char, str(index) + ".png")
                mkdir(save_path)
                cv2.imencode(".png", image)[1].tofile(save_path)

    def get_image_dict(self):
        return self.image_dict


class ImagePreprocessorConfig(ConfigTemplate):
    def __init__(self,
                 dataset_file_root,
                 character_list,
                 augmentation_amount,
                 out_image_size,
                 out_file_root,
                 save_image_format_result=False
                 ):
        self.DATASET_FILE_ROOT = dataset_file_root
        self.CHARACTER_LIST = character_list
        self.AUGMENTATION_AMOUNT = augmentation_amount
        self.OUT_IMAGE_SIZE = out_image_size
        self.OUT_FILE_ROOT = out_file_root
        self.SAVE_IMAGE_FORMAT_RESULT = save_image_format_result


# only test case
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    my_conf = ImagePreprocessorConfig(
        dataset_file_root="dataset/character_data_46x90/",
        character_list=[str(i) for i in range(10)] + list("QWERTYUPASDFGHJKLZXCVBNM") + [
            '冀', '新', '鄂', '宁', '桂', '黑', '湘', '皖', '云', '豫', '蒙', '赣', '吉', '辽', '苏', '甘', '晋', '浙', '闽',
            '渝', '贵', '陕', '粤', '川', '鲁', '琼', '青', '藏', '京', '津', '沪'],
        augmentation_amount=100,
        out_image_size=(13 * 5, 25 * 5, 3),
        out_file_root="dataset/character_data_augmentation/",
        save_image_format_result=True
    )

    my_preprocessor = ImagePreprocessor(my_conf)
