import os
from fastapi import UploadFile, File
from celery import chain

from .. import worker
from . import app, utils, paths


@app.get('/')
async def index():
    return 'ok'


@app.post('/predictStrawberriesBoundingBoxesAndSegmentation')
def predict_strawberries_bboxes(image: UploadFile = File(...)):
    file_name, image_path = utils.generate_file_name_and_path(image)
    with open(image_path, 'wb') as f:
        f.write(image.file.read())
    bboxes_image_path = os.path.join(paths.MEDIA_DIR, 'bboxes', file_name)
    segmentation_image_path = os.path.join(paths.MEDIA_DIR, 'segmentation', file_name)
    result = chain(
        worker.get_plant_segmentation_masks.s(image_path, segmentation_image_path),
        worker.get_strawberries_bboxes.s(image_path, bboxes_image_path)
    )()
    bboxes = result.get()
    return {
        'imgUrl': f'/images/initial/{file_name}',
        'bboxesImgUrl': f'/images/bboxes/{file_name}',
        'segmentationImgUrl': f'/images/segmentation/{file_name}',
        'bboxes': bboxes,
        # 'segmentationMasks': result.parent.get(),
    }


@app.post('/predictPlantDiseases')
def predict_plant_diseases(image: UploadFile = File(...)):
    file_name, image_path = utils.generate_file_name_and_path(image)
    with open(image_path, 'wb') as f:
        f.write(image.file.read())
    result = worker.get_plant_diagnose.s(image_path).apply_async()
    return result.get()


@app.post('/predictPlantLevel')
def predict_plant_level(image: UploadFile = File(...)):
    file_name, image_path = utils.generate_file_name_and_path(image)
    with open(image_path, 'wb') as f:
        f.write(image.file.read())
    result = worker.get_plant_level.s(image_path).apply_async()
    return result.get()
