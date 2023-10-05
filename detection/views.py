from django.shortcuts import render
from .forms import UploadImageForm
from .models import ASLMODEL
from PIL import Image
import io
import numpy as np

# Create your views here.
def home(request):
    return(render(request,'detection/upload.html'))

def process_image(request):
    if request.method == "POST":
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.cleaned_data['image']
            file_buffer = io.BytesIO(f.read())

            # Open the image using Pillow (PIL)
            pil_image = Image.open(file_buffer) 
            model = ASLMODEL.objects.get(active=True)
            pr_image = model.preprocess_image(pil_image)
            prediction = model.predict(pr_image)
            return render(request, 'detection/upload.html', {'prediction': prediction})
    else:
        form = UploadImageForm()
    return render(request, 'detection/upload.html', {'form': form})
