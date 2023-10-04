from django.db import models
from django.utils import timezone
from tensorflow import keras
import numpy as np
import pandas as pd
import json
from django.conf import settings
import os
from PIL import Image
# Create your models here.

class ASLMODEL(models.Model):
    name = models.CharField(max_length=50)
    path = models.CharField(max_length=200)
    created_on = models.DateTimeField(default=timezone.now)
    active = models.BooleanField()
    configmap = models.TextField(blank=True, null=True, default=None)

    def activate(self):
        self.active = True
        self.save()

    def deactivate(self):
        self.active = False
        self.save()

    def __str__(self):
        return self.name
    
    def get_path(self):
        return self.path
    
    def load_model(self):
        model_path = os.path.join(settings.BASE_DIR, self.path)
        model = keras.models.load_model(model_path)
        return model
 
    def predict(self, image_array):
        model = self.load_model()
        configmap = json.loads(self.configmap)
        if image_array.shape == (1, 224, 224, 3):
            predictions = model.predict(image_array)
            prediction = configmap[str(predictions.argmax())]
        else:
            prediction = None
        return prediction
    
    def preprocess_image(self, image_name):
        image_path = os.path.join(settings.BASE_DIR, 'DATA', 'Test', image_name)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = image_array / 255.0
        return image_array.reshape((-1, 224, 224, 3))
