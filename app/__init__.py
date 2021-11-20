import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

from . import paths
from .dto import StrawberryPredictions
from .models_service import models_service

app = FastAPI(root_path='/api')
app.mount('/images', StaticFiles(directory=paths.MEDIA_DIR), name='images')


@app.get('/')
async def index():
    return 'ok'


@app.post('/predict')
def predict(image: UploadFile = File(...)):
    filename = f'{uuid4()}.{image.filename.rsplit(".", 1)[-1]}'
    result_image, preds = models_service.get_strawberries_bboxes(image.file)
    result_image.save(os.path.join(paths.MEDIA_DIR, filename))
    return {
        'renderedImgUrl': f'/images/{filename}',
        'predictions': preds,
    }
