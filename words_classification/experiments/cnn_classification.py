import os
import cv2
import logging

import numpy as np

from data_loaders.image_preprocessor import ImagePreprocessor, ImagePreprocessorConfig
from data_loaders.image_set_loader import ImageSetDataLoader, ImageSetDataLoaderConfig
from models.cnn_model import CNNModel, CNNModelConfig
from trainers.universal_trainer import UniversalTrainer, UniversalTrainerConfig

logger = logging.getLogger(__name__)

IS_TRAINING = True
INPUT_SHAPE = (65, 125, 1)
INPUT_SHAPE_REVERSE = (125, 65, 1)
CHARACTER_LIST = [str(i) for i in range(10)] + list("QWERTYUPASDFGHJKLZXCVBNM") + [
    '冀', '新', '鄂', '宁', '桂', '黑', '湘', '皖', '云', '豫', '蒙', '赣', '吉', '辽', '苏', '甘', '晋', '浙', '闽',
    '渝', '贵', '陕', '粤', '川', '鲁', '琼', '青', '藏', '京', '津', '沪']

preprocessor_config = ImagePreprocessorConfig(
    dataset_file_root="dataset/character_data_46x90/",
    character_list=CHARACTER_LIST,
    augmentation_amount=1000,
    out_image_size=INPUT_SHAPE,
    out_file_root="dataset/character_data_augmentation/"
)

data_loader_config = ImageSetDataLoaderConfig(
    pickle_dumped_file="dataset/character_data_augmentation/dataset_dict_dump.pickle",
    image_shape=INPUT_SHAPE_REVERSE,
    shuffle_buffer_size=100000,
    batch_size=10
)

model_config = CNNModelConfig(input_shape=INPUT_SHAPE_REVERSE,
                              output_len=len(CHARACTER_LIST),
                              learning_rate=0.001)

trainer_config = UniversalTrainerConfig(epoch=5)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    if IS_TRAINING:
        logger.info("Preprocess...")
        preprocessor = ImagePreprocessor(preprocessor_config)
        logger.info("Making dataset...")
        data_loader = ImageSetDataLoader(data_loader_config)

        model = CNNModel(model_config)
        trainer = UniversalTrainer(model.get_model(), data_loader.get_dataset(), trainer_config)

        logger.info("Training...")
        trainer.train()
        trainer.save("log/test.h5")
    else:
        model = CNNModel(model_config)
        trainer = UniversalTrainer(model.get_model(), None, trainer_config)
        trainer.load("log/test.h5")

        VALIDATION_IMAGE_ROOT = "dataset/validation_image/"

        valid_image_list = os.listdir(VALIDATION_IMAGE_ROOT)
        logger.debug("Validation image list: %s", valid_image_list)

        for image_file_name in valid_image_list:
            image_file = os.path.join(VALIDATION_IMAGE_ROOT, image_file_name)
            image = cv2.imread(image_file, flags=cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, INPUT_SHAPE[:2])
            image = np.array([image, ]) / 255
            image = np.reshape(image, np.shape(image) + (1,))

            predict_result = np.argmax(trainer.model.predict(image, batch_size=1), axis=-1)[0]

            logger.info("Predict result: %s %s",
                        predict_result,
                        CHARACTER_LIST[predict_result], )

            cv2.imshow("Validation image", image[0])
            cv2.waitKey(1000)
