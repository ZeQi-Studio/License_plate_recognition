import logging

from data_loaders.image_preprocessor import ImagePreprocessor, ImagePreprocessorConfig
from data_loaders.image_set_loader import ImageSetDataLoader, ImageSetDataLoaderConfig
from models.cnn_model import CNNModel, CNNModelConfig
from trainers.universal_trainer import UniversalTrainer, UniversalTrainerConfig

logger = logging.getLogger(__name__)

INPUT_SHAPE = (65, 125, 3)
INPUT_SHAPE_REVERSE = (125, 65, 3)
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
    shuffle_buffer_size=10000,
    batch_size=10
)

model_config = CNNModelConfig(input_shape=INPUT_SHAPE_REVERSE,
                              output_len=len(CHARACTER_LIST),
                              learning_rate=0.001)

trainer_config = UniversalTrainerConfig(epoch=3)

if __name__ == '__main__':
    # preprocessor = ImagePreprocessor(preprocessor_config)
    data_loader = ImageSetDataLoader(data_loader_config)
    model = CNNModel(model_config)
    trainer = UniversalTrainer(model.get_model(), data_loader.get_dataset(), trainer_config)
    trainer.train()
