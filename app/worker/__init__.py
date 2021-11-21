from celery import Celery
from kombu.serialization import register

from . import serializer, config, models_services as ms

# register(
#     'custom-json',
#     serializer.dumps,
#     serializer.loads,
#     content_type='application/x-custom-json',
#     content_encoding='utf-8'
# )

app = Celery('tasks')
app.config_from_object(config)


@app.task
def get_strawberries_bboxes(segmentation_masks, image_path, result_image_path):
    # print('Strawberries bboxes task started')
    # print(segmentation_masks['berries'].shape)
    return ms.maturity_service.get_strawberries_bboxes(
        image_path,
        result_image_path,
        segmentation_masks['berries']
    )


@app.task
def get_plant_diagnose(image_path):
    return ms.diagnose_service.diagnose(image_path)


@app.task
def get_plant_segmentation_masks(image_path, result_image_path):
    return ms.segmentation_service.segment(image_path, result_image_path)


@app.task
def get_plant_level(image_path):
    return {
        'level': ms.level_service.predict_level(image_path),
    }
