# TODO: install this dependency: pip install -U segmentation-models-pytorch
# TODO: install albumentations
import io
import os
import albumentations as albu
from torchvision.utils import draw_segmentation_masks
import segmentation_models_pytorch as smp
import torch
import numpy as np
from cv2 import cv2

import matplotlib.pyplot as plt
from PIL import Image

from app import paths


class SegmentationService:
    def __init__(self, weights_pth: str):
        self._model = None
        self.weights_pth = weights_pth

        self.ENCODER = 'se_resnext50_32x4d'
        self.ENCODER_WEIGHTS = 'imagenet'
        self.CLASSES = ["berry", "leaf", "stem", "flower"]
        self.ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
        self.DEVICE = 'cuda'
        self.preprocessing_fn = _get_preprocessing(
            smp.encoders.get_preprocessing_fn(self.ENCODER, self.ENCODER_WEIGHTS)
        )

    @property
    def model(self):
        if self._model is None:
            self._model = smp.FPN(
                encoder_name=self.ENCODER,
                encoder_weights=self.ENCODER_WEIGHTS,
                classes=len(self.CLASSES),
                activation=self.ACTIVATION,
            )
            self._model.load_state_dict(torch.load(self.weights_pth))
            self._model = self._model.to(torch.device(self.DEVICE))
        return self._model

    def segment(self, inp_pth: str, out_pth: str) -> np.array:
        # TODO: where to store berries mask?
        image = np.array(Image.open(inp_pth))
        model_inp = self._preproc_image(image)
        model_inp = torch.from_numpy(model_inp).to(self.DEVICE).unsqueeze(0)

        pr_mask = self.model.predict(model_inp)
        pr_mask = pr_mask.squeeze().cpu().numpy().round()

        img_tensor = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.uint8)
        pr_mask = np.stack(
            [cv2.resize(pr_mask[idx], dsize=(img_tensor.shape[2], img_tensor.shape[1]))
             for idx in range(len(self.CLASSES))]
        )

        segm_img = draw_segmentation_masks(img_tensor, torch.tensor(pr_mask, dtype=torch.bool), alpha=0.3,
                                           colors=["blue", "red", "purple", "green"])

        # segm_img = segm_img.numpy().astype(int)

        plt.imshow(np.transpose(segm_img.numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.savefig(out_pth, bbox_inches='tight')

        # # the GK's data
        # strawberry_slice = pr_mask[1]
        # Image.fromarray(segm_img).save(out_pth)
        # lst_img = segm_img.tolist()
        # Image.fromarray(np.array(lst_img)).save(out_pth)
        # print("out_pth", out_pth, "segm_img", segm_img.shape)
        # cv2.imwrite(out_pth, segm_img)

        return {
            "leaves": pr_mask[0].tolist(),
            "berries": pr_mask[1].tolist(),
            "stems": pr_mask[2].tolist(),
            "flowers": pr_mask[3].tolist()
        }

    def _preproc_image(self, image: np.array) -> np.array:
        image = cv2.resize(image, dsize=(512, 512))

        mask = np.zeros((512, 512, len(self.CLASSES)), dtype=float)
        sample = self.preprocessing_fn(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']

        return image


def _to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def _get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=_to_tensor, mask=_to_tensor),
    ]
    return albu.Compose(_transform)


segmentation_service = SegmentationService(os.path.join(paths.MODELS_DIR, 'segmentation_weights.pt'))
