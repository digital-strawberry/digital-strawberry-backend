from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import paths

app = FastAPI(root_path='/api')
app.mount('/images', StaticFiles(directory=paths.MEDIA_DIR), name='images')

from . import routes
