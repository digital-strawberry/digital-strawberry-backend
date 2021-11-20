import os
from uuid import uuid4
from fastapi import UploadFile

from app import paths


def generate_file_name_and_path(image: UploadFile):
    image_id = uuid4()
    file_name = f'{image_id}.{image.filename.rsplit(".", 1)[-1]}'
    image_path = os.path.join(paths.MEDIA_DIR, 'initial', file_name)
    return file_name, image_path
