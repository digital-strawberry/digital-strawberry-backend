import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
import torch

from .dto import StrawberryPredictions

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR, 'media')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# models
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=os.path.join(MODELS_DIR, 'yolo_weights.pt'),
    force_reload=True
).autoshape()

app = FastAPI()
app.mount('/images', StaticFiles(directory=MEDIA_DIR), name='images')


@app.get('/')
async def index():
    return 'hello world'


@app.post('/predict')
def predict(image: UploadFile = File(...)):
    filename = f'{uuid4()}.{image.filename.rsplit(".", 1)[-1]}'
    # with open(os.path.join(MEDIA_DIR, filename), 'wb') as f:
    #     f.write(image.file.read())
    res = model(image.file)
    res.save(os.path.join(MEDIA_DIR, filename))
    return {
        'url': f'/images/{filename}',
        'health': 100,
        'entities': [
            {
                'x': 100,
                'y': 100,
                'width': 300,
                'height': 300,
            }
        ]
    }
