
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

UPLOAD_FOLDER = './upload'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'style-image' not in request.files:
            return 'there is no file1 in form!'
        if 'content-image' not in request.files:
            return 'there is no file2 in form!'
        file1 = request.files['style-image']
        file2 = request.files['content-image']
        path1 = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path1)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
        file2.save(path2)
        # return path
        stylized_image = model(
            tf.constant(
                load_image(
                    path1
                )
            ),
            tf.constant(
                load_image(
                    path2
                )
            )
        )[0]
        plt.imsave('static/style.png', np.squeeze(stylized_image))
        return '''
        <img src='style.png' />
        '''
    return render_template("submit.html")


if __name__ == "__main__":
    model = tf.saved_model.load("./model/")
    app.run(host='0.0.0.0', port=8080, debug=True)
