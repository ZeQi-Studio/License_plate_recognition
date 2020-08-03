import logging
import json
import base64
import cv2

import numpy as np

from flask import Flask, request
from flask_cors import cross_origin

from license_plate_crop.License_plate_detection_and_cut import detection_and_cut_from_array
from words_classification.models.cnn_model import CNNModel, CNNModelConfig

# variable
INPUT_SHAPE = (65, 125, 1)
INPUT_SHAPE_REVERSE = (125, 65, 1)
CHARACTER_LIST = [str(i) for i in range(10)] + list("QWERTYUPASDFGHJKLZXCVBNM") + [
    '冀', '新', '鄂', '宁', '桂', '黑', '湘', '皖', '云', '豫', '蒙', '赣', '吉', '辽', '苏', '甘', '晋', '浙', '闽',
    '渝', '贵', '陕', '粤', '川', '鲁', '琼', '青', '藏', '京', '津', '沪']

# config for DL model
model_config = CNNModelConfig(input_shape=INPUT_SHAPE_REVERSE,
                              output_len=len(CHARACTER_LIST),
                              learning_rate=0.001)

logger = logging.getLogger(__name__)
app = Flask(__name__)


def decode_base64_image(image_in_base64):
    """
    from base64 string to np.array
    :param image_in_base64:
    :return:
    """
    b_image = base64.b64decode(image_in_base64)
    return cv2.imdecode(np.frombuffer(b_image, np.uint8), cv2.IMREAD_COLOR)


@app.route('/')
def hello_world():
    return 'hello_world'


# test case
@app.route('/detect', methods=['POST'])
@cross_origin()
def detect():
    model = CNNModel(model_config)
    model.load("words_classification/log/test.h5")

    logger.debug("Header: %s", request.headers)
    # logger.debug("Form: %s", request.form)

    image = decode_base64_image(request.form.get("image"))
    logger.debug("Image after decode shape: %s", np.shape(image))
    char_image_array = detection_and_cut_from_array(image)[1:]
    logger.debug("Image crop result shape: %s", np.shape(char_image_array[0]))

    for image in char_image_array:
        cv2.imshow("after crop", image)
        cv2.waitKey(1000)

    char_image_array = [np.reshape(cv2.resize(x, (65, 125)), (125, 65, 1)) for x in char_image_array]
    char_image_array = np.array(char_image_array) / 255
    logger.debug("Image resize result shape: %s", np.shape(char_image_array))

    # for image in char_image_array:
    #     cv2.imshow("after resize", image)
    #     cv2.waitKey()

    license_plate_str = ""
    if char_image_array is not None:
        predict_result = np.argmax(model.get_model().predict(char_image_array), axis=-1)
        logger.debug("Predict result: %s", predict_result)
        for char_result in predict_result:
            license_plate_str += CHARACTER_LIST[char_result]

    logger.info("Predict result: %s", license_plate_str)

    return json.dumps({"result": license_plate_str})


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="0.0.0.0", debug=True)
