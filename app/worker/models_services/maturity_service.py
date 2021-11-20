import os
import torch
from PIL import Image
from typing import BinaryIO
from cv2 import cv2
import collections

from app import paths


class MaturityService:
    def __init__(self):
        self._yolo = None

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

    def get_strawberries_maturity(self, img, preds):
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for pred in preds:
            cropped_strawberry = img[pred['ymin']:pred['ymax'], pred['xmin']:pred['xmax']]

            hsv_strawberry = cv2.cvtColor(cropped_strawberry, cv2.COLOR_RGB2HSV)
            red1 = cv2.inRange(hsv_strawberry[:, :, 0], 170, 180)
            red2 = cv2.inRange(hsv_strawberry[:, :, 0], 0, 10)
            green = cv2.inRange(hsv_strawberry[:, :, 0], 28, 80)
            red = cv2.bitwise_or(red1, red2)
            stat = sum(collections.Counter(x)[255] for x in red)
            total = red.shape[0] * red.shape[1] - sum(collections.Counter(x)[255] for x in green)
            if total == 0:
                percentage = 0.0
            else:
                percentage = round(stat / total / 0.8, 3)
            if percentage > 1:
                percentage = 1.0
            cv2.rectangle(img, (pred['xmin'], pred['ymin']), (pred['xmax'], pred['ymax']), (255, 165, 0), 1)

            text = f'Maturity {percentage}'
            img = self.draw_bb(img, pred, text)
            pred['maturity'] = percentage
        return img

    def get_strawberries_bboxes(self, file_path: str, result_file_path: str) -> [Image, list[dict]]:
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
        img = self.get_strawberries_maturity(res.imgs[0], processed_preds)
        Image.fromarray(img).save(result_file_path)
        return processed_preds


maturity_service = MaturityService()
