from celery import Celery

from .models_services import diagnose_service, maturity_service

app = Celery('tasks', broker='redis://localhost', backend='redis://localhost')

@app.task
def get_strawberries_bboxes(image_path, result_image_path):
    return maturity_service.get_strawberries_bboxes(image_path, result_image_path)

@app.task
def get_plant_diagnose(image_path):
    return diagnose_service.diagnose(image_path)
