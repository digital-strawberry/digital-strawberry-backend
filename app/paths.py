import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR, 'media')
# create ./media if doesn't exist
if not os.path.exists(MEDIA_DIR):
    os.makedirs(MEDIA_DIR)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
