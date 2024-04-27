from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
import numpy as np
from PIL import Image
import os
import mimetypes

app = Flask(__name__)

# Define the U-Net model architecture
def build_unet_model():
    input_layer = Input(shape=(128, 128, 3))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    up1 = UpSampling2D((2, 2))(conv3)
    concat1 = concatenate([conv2, up1], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    up2 = UpSampling2D((2, 2))(conv4)
    concat2 = concatenate([conv1, up2], axis=-1)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(concat2)
    model = Model(inputs=input_layer, outputs=outputs)
    return model

# Load the trained U-Net model
unet_model = build_unet_model()
unet_model.load_weights('unet_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    tumor_present = None
    image_path = None
    mask_path = None
    if request.method == 'POST':
        try:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            upload_folder = os.path.join(basepath, 'uploads')
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            filename = secure_filename(f.filename)
            file_path = os.path.join(upload_folder, filename)
            f.save(file_path)

            image = Image.open(file_path)
            image_array = preprocess_image(image)
            prediction = unet_model.predict(image_array)
            threshold = 0.5
            prediction_binary = (prediction > threshold).astype(np.uint8)
            tumor_present = np.any(prediction_binary == 1)
            
            image_path = os.path.join('/uploads', filename)
            mask_path = '/static/mask.png'
            save_mask(prediction_binary[0], os.path.join(basepath, 'static', 'mask.png'))

        except Exception as e:
            print(str(e))
            return str(e), 500

    return render_template('upload.html', tumor_present=tumor_present, image_path=image_path, mask_path=mask_path)

def preprocess_image(image):
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def save_mask(mask, mask_path):
    mask = np.squeeze(mask, axis=-1)
    mask = Image.fromarray(mask * 255)
    mask = mask.convert('RGB')
    mask.save(mask_path)

if __name__ == '__main__':
    app.run(debug=True)
