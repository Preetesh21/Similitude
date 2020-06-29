import os
import sys
import numpy as np
import cv2
import json


# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.models import load_model
from keras.preprocessing import image

from util import base64_to_pil

# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/


print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'C:\\Users\\verma\\Downloads\\fashion_similarity_model (1).h5'

# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')
model=load_model(MODEL_PATH)

def model_predict(img1,img2, model):
    img1=np.array(img1)
    img2=np.array(img2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    gray11=cv2.resize(gray1, (28, 28),  
               interpolation = cv2.INTER_AREA)

    gray22=cv2.resize(gray2, (28, 28),  
               interpolation = cv2.INTER_AREA)

    a=np.array(gray11)
    b=np.array(gray22)

    a=a.reshape(1,28,28,1)
    b=b.reshape(1,28,28,1)


    result=model.predict([a,b])
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        print(request.json)
        print('aaa')
        img1,img2 = base64_to_pil(request.json)
        
        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img1,img2, model)

        # Process your result for human
        # pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        

        # result = str(pred_class[0][0][1])               # Convert to string
        # result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        print(preds)
        type(preds)
        result=preds
        lists = result.tolist()
        json_str = json.dumps(lists)
        print(json_str)
        return jsonify(json_str)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
