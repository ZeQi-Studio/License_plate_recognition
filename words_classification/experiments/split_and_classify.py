import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error
import cv2
import logging

import numpy as np

from data_loaders.image_preprocessor import ImagePreprocessor, ImagePreprocessorConfig
from data_loaders.image_set_loader import ImageSetDataLoader, ImageSetDataLoaderConfig
from models.cnn_model import CNNModel, CNNModelConfig
from trainers.universal_trainer import UniversalTrainer, UniversalTrainerConfig

logger = logging.getLogger(__name__)

IS_TRAINING = False
DATA_AUGMENTATION = True and IS_TRAINING

INPUT_SHAPE = (65, 125, 1)
INPUT_SHAPE_REVERSE = (125, 65, 1)

VALIDATION_IMAGE_CHAR_ROOT = "dataset/validation_image_char/"
VALIDATION_IMAGE_PROVINCE_ROOT = "dataset/validation_image_province/"

# split word classify
CHARACTER_LIST = [str(i) for i in range(10)] + list("QWERTYUPASDFGHJKLZXCVBNM")
PROVINCE_LIST = ['冀', '新', '鄂', '宁', '桂', '黑', '湘', '皖', '云', '豫', '蒙', '赣', '吉', '辽', '苏', '甘', '晋', '浙', '闽',
                 '渝', '贵', '陕', '粤', '川', '鲁', '琼', '青', '藏', '京', '津', '沪']

preprocessor_char_config = ImagePreprocessorConfig(
    dataset_file_root="dataset/character_data_46x90/",
    character_list=CHARACTER_LIST,
    augmentation_amount=500,
    out_image_size=INPUT_SHAPE,
    out_file_root="dataset/character_data_augmentation/",
    save_image_format_result=False
)

preprocessor_province_config = ImagePreprocessorConfig(
    dataset_file_root="dataset/character_data_46x90/",
    character_list=PROVINCE_LIST,
    augmentation_amount=500,
    out_image_size=INPUT_SHAPE,
    out_file_root="dataset/character_data_augmentation/",
    save_image_format_result=False
)

model_char_config = CNNModelConfig(input_shape=INPUT_SHAPE_REVERSE,
                                   output_len=len(CHARACTER_LIST),
                                   learning_rate=0.0001)
model_province_config = CNNModelConfig(input_shape=INPUT_SHAPE_REVERSE,
                                       output_len=len(PROVINCE_LIST),
                                       learning_rate=0.0001)

trainer_config = UniversalTrainerConfig(epoch=5)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    if IS_TRAINING:
        logger.info("Preprocess...")
        preprocessor_char = ImagePreprocessor(preprocessor_char_config)
        preprocessor_province = ImagePreprocessor(preprocessor_province_config)
        logger.info("Making dataset...")
        data_loader_char_config = ImageSetDataLoaderConfig(
            image_dict=preprocessor_char.get_image_dict(),
            image_shape=INPUT_SHAPE_REVERSE,
            shuffle_buffer_size=100000,
            batch_size=100
        )
        data_loader_province_config = ImageSetDataLoaderConfig(
            image_dict=preprocessor_province.get_image_dict(),
            image_shape=INPUT_SHAPE_REVERSE,
            shuffle_buffer_size=100000,
            batch_size=100
        )

        data_loader_char = ImageSetDataLoader(data_loader_char_config)
        data_loader_province = ImageSetDataLoader(data_loader_province_config)

        model_char = CNNModel(model_char_config)
        model_province = CNNModel(model_province_config)

        trainer_char = UniversalTrainer(model_char.get_model(),
                                        data_loader_char.get_dataset(),
                                        trainer_config)
        trainer_province = UniversalTrainer(model_province.get_model(),
                                            data_loader_province.get_dataset(),
                                            trainer_config)

        logger.info("Training...")
        trainer_char.train()
        trainer_province.train()
        trainer_char.save("log/test_split_char.h5")
        trainer_province.save("log/test_split_province.h5")
    else:
        model_char = CNNModel(model_char_config)
        trainer_char = UniversalTrainer(model_char.get_model(), None, trainer_config)
        trainer_char.load("log/test_split_char.h5")

        model_province = CNNModel(model_province_config)
        trainer_province = UniversalTrainer(model_province.get_model(), None, trainer_config)
        trainer_province.load("log/test_split_province.h5")

        # province
        valid_image_province_list = os.listdir(VALIDATION_IMAGE_PROVINCE_ROOT)
        logger.debug("Validation image list: %s", valid_image_province_list)
        for image_file_name in valid_image_province_list:
            image_file = os.path.join(VALIDATION_IMAGE_PROVINCE_ROOT, image_file_name)
            image = cv2.imread(image_file, flags=cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 0)
            image = cv2.resize(image, INPUT_SHAPE[:2])
            image = np.array([image, ]) / 255
            image = np.reshape(image, np.shape(image) + (1,))
            predict_result = np.argmax(trainer_char.model.predict(image, batch_size=1), axis=-1)[0]

            logger.info("Predict result: %s %s",
                        predict_result,
                        PROVINCE_LIST[predict_result], )

            cv2.imshow("Validation image", image[0])
            cv2.waitKey()

        # char
        valid_image_char_list = os.listdir(VALIDATION_IMAGE_CHAR_ROOT)
        logger.debug("Validation image list: %s", valid_image_char_list)
        for image_file_name in valid_image_char_list:
            image_file = os.path.join(VALIDATION_IMAGE_CHAR_ROOT, image_file_name)
            image = cv2.imread(image_file, flags=cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 0)
            image = cv2.resize(image, INPUT_SHAPE[:2])
            image = np.array([image, ]) / 255
            image = np.reshape(image, np.shape(image) + (1,))
            predict_result = np.argmax(trainer_char.model.predict(image, batch_size=1), axis=-1)[0]

            logger.info("Predict result: %s %s",
                        predict_result,
                        CHARACTER_LIST[predict_result], )

            cv2.imshow("Validation image", image[0])
            cv2.waitKey()
