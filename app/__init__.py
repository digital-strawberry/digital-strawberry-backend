import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

from .dto import StrawberryPredictions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR, 'media')

app = FastAPI()
app.mount('/images', StaticFiles(directory=MEDIA_DIR), name='images')


@app.get('/')
async def index():
    return 'hello world'


@app.post('/predict')
def predict(image: UploadFile = File(...)):
    filename = f'{uuid4()}.{image.filename.rsplit(".", 1)[-1]}'
    with open(os.path.join(MEDIA_DIR, filename), 'wb') as f:
        f.write(image.file.read())
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
