import logging
from flask import Flask, request

logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'hello_world'


# test case
@app.route('/detect', methods=['POST'])
def detect():
    logger.debug("Header: %s", request.headers)
    logger.debug("Form: %s", request.form)

    return "The image I receive is: " + str(request.form.get('image'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="0.0.0.0", debug=True)
