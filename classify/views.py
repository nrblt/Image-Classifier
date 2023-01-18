from django.shortcuts import render
from django.http import JsonResponse
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
from tensorflow.python.keras.backend import set_session
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import vgg16
import datetime
import traceback
from classify.models import Image

def index(request):
    k = request.session.get('k')
    if k is None:
        k=1

    if request.method == "POST":
        data = {"success": False}
        f = request.FILES['sentFile']  # here you get the files needed
        response = {}
        file_name = "pic"
        file_name = file_name + str(k)

        file_name = file_name + ".jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = '/Users/macbook/PycharmProjects/image-classification/media/pic' + str(k) + '.jpg'
        k = k + 1
        request.session['k'] = k
        original = load_img(file_url, target_size=(224, 224))
        numpy_image = img_to_array(original)
        print(file_url)
        image_batch = np.expand_dims(numpy_image, axis=0)
        # prepare the image for the VGG model
        processed_image = vgg16.preprocess_input(image_batch.copy())

        set_session(settings.SESS)
        predictions = settings.VGG_MODEL.predict(processed_image)

        label = decode_predictions(predictions)
        print(label)

        show = []
        for i in range(0, min(5, len(label[0]))):
            mytup=(label[0][i][1],"{:.2f}".format(label[0][i][2] * 100))
            show.append(mytup)
            dbImage = Image()
            dbImage.path = file_url
            dbImage.predicted_label = str('%s (%.2f%%)' % (label[0][i][1], label[0][i][2] * 100))
            dbImage.save()

        response['names'] = show
        return render(request, 'answerpage.html', response)
    else:
        return render(request, 'homepage.html')
