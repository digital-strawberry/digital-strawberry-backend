import os

import numpy as np
import torch
from PIL import Image
from cv2 import cv2
import collections

from app import paths


def _convert_gray2rgb(image):
    width, height = image.shape
    out = np.empty((width, height, 3), dtype=np.uint8)
    out[:, :, 0] = image
    out[:, :, 1] = image
    out[:, :, 2] = image
    return out


class MaturityService:
    def __init__(self):
        self._yolo = None

    gray = np.random.rand(256, 256)  # gray scale image
    gray2rgb = _convert_gray2rgb(gray)
    # load yolo model lazily for graphic memory saving
    @property
    def yolo(self):
        if self._yolo is None:
            self._yolo = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path=os.path.join(paths.MODELS_DIR, 'yolo_weights.pt'),
            )
        return self._yolo

    def draw_bb(self, img, pred, label):
        p1, p2 = (pred['xmin'], pred['ymin']), (pred['xmin'], pred['ymin'])
        cv2.rectangle(img, p1, p2, (255, 165, 0), thickness=2, lineType=cv2.LINE_AA)
        tf = 1
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, (255, 165, 0), -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 2 / 3, (255, 255, 255),
                    thickness=tf, lineType=cv2.LINE_AA)
        return img

    def get_strawberries_maturity(self, img, preds, berries_mask):
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        berries_mask = np.asarray(berries_mask)
        for pred in preds:
            cropped_strawberry = img[pred['ymin']:pred['ymax'], pred['xmin']:pred['xmax']]
            cropped_mask_1d = berries_mask[pred['ymin']:pred['ymax'], pred['xmin']:pred['xmax']]
            cropped_mask_3d = _convert_gray2rgb(cropped_mask_1d)
            only_berry = np.multiply(cropped_strawberry, cropped_mask_3d)
            estimate_sum = 0
            hsv_strawberry = cv2.cvtColor(only_berry, cv2.COLOR_RGB2HSV)
            for x in range(hsv_strawberry.shape[0]):
                for y in range(hsv_strawberry.shape[1]):
                    h, s, v = hsv_strawberry[x, y]
                    if v <= 20:
                        continue
                    if s <= 40:
                        estimate_sum += 0.5
                    elif 170 <= h <= 180 or 0 <= h <= 10:
                        estimate_sum += 1
            estimate_sum /= (hsv_strawberry.shape[0] * hsv_strawberry.shape[1])
            percentage = round(estimate_sum / 0.95, 3)
            cv2.rectangle(img, (pred['xmin'], pred['ymin']), (pred['xmax'], pred['ymax']), (255, 165, 0), 1)

            text = f'Maturity {percentage}'
            img = self.draw_bb(img, pred, text)
            pred['maturity'] = percentage
        return img

    def get_strawberries_bboxes(
            self,
            file_path: str,
            result_file_path: str,
            berries_mask: np.ndarray
    ) -> [Image, list[dict]]:
        pil_image = Image.open(file_path)
        res = self.yolo(pil_image)
        # res.render()
        processed_preds = []
        for cors in res.xyxy[0]:
            processed_preds.append({
                'xmin': int(cors.data[0]),
                'ymin': int(cors.data[1]),
                'xmax': int(cors.data[2]),
                'ymax': int(cors.data[3]),
                'confidence': cors.data[4].item(),
            })
        img = self.get_strawberries_maturity(res.imgs[0], processed_preds, berries_mask)
        Image.fromarray(img).save(result_file_path)
        return processed_preds


maturity_service = MaturityService()
