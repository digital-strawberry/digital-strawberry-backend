import os
from fastapi import UploadFile, File

from .. import worker
from . import app, utils, paths


@app.get('/')
async def index():
    return 'ok'


@app.post('/predictStrawberriesBoundingBoxes')
def predict_strawberries_bboxes(image: UploadFile = File(...)):
    file_name, image_path = utils.generate_file_name_and_path(image)
    with open(image_path, 'wb') as f:
        f.write(image.file.read())
    result_image_path = os.path.join(paths.MEDIA_DIR, 'processed', file_name)
    result = worker.get_strawberries_bboxes.apply_async((image_path, result_image_path))
    preds = result.get()
    return {
        'imgUrl': f'/images/initial/{file_name}',
        'renderedImgUrl': f'/images/processed/{file_name}',
        'predictions': preds,
    }


@app.post('/predictPlantDiseases')
def predict_plant_diseases(image: UploadFile = File(...)):
    file_name, image_path = utils.generate_file_name_and_path(image)
    with open(image_path, 'wb') as f:
        f.write(image.file.read())
    result = worker.get_plant_diagnose.apply_async((image_path, ))
    return result.get()
