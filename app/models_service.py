import os
import torch
from PIL import Image
from typing import BinaryIO

from . import paths


class ModelsService:
    yolo = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=os.path.join(paths.MODELS_DIR, 'yolo_weights.pt'),
    )

    def get_strawberries_bboxes(self, image: BinaryIO) -> [Image, list[dict]]:
        pil_image = Image.open(image)
        res = self.yolo(pil_image)
        res.render()
        processed_preds = []
        for cors in res.xyxy[0]:
            cors = list(map(int, cors.data))
            processed_preds.append({
                'xmin': cors[0],
                'ymin': cors[1],
                'xmax': cors[2],
                'ymax': cors[3],
                'confidence': cors[4],
            })
        return Image.fromarray(res.imgs[0]), processed_preds


models_service = ModelsService()
