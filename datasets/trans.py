import random

from PIL import ImageFilter
import albumentations as A
from albumentations.augmentations.crops import random_crop
import cv2
import numpy as np

from datasets.pipelines import transforms as Tmm
import datasets.transforms as T


def make_self_det_transforms(image_set, res, isjitter=False):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    assert res in [480, 800], f'{res} is not supported only 480 or 800'

    # The image of ImageNet is relatively small.
    if res == 480:
        scales = [320, 336, 352, 368, 400, 416, 432, 448, 464, 480]
        min_size = 480
        max_size = 800
    else:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        min_size = 800
        max_size = 1333

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            T.RandomApply(
                T.ColorJitter(brightness=0.4, contrast=0.5,
                              saturation=0.5, hue=0.1),
                prob=0.8 if isjitter else 0
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([min_size], max_size=max_size),
            normalize,
        ])

    if image_set == 'teacher':
        return T.Compose([
            # T.CenterCrop(size=(720,404)),
            T.RandomResize([min_size], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def teacher_transforms_albu(res):
    assert res in [480, 800], f'{res} is not supported only 480 or 800'

    # The image of ImageNet is relatively small.
    if res == 480:
        scales = [320, 336, 352, 368, 400, 416, 432, 448, 464, 480]
        min_size = 480
        max_size = 900
    else:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        min_size = 800
        max_size = 1333
    trans = A.Compose([
        A.HorizontalFlip(p=0.5),
        BoxSafeCrop(min_size, min_size, erosion_rate=0),
        A.OneOf(
            [A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, p=1)], p=0.2
        ),
        A.OneOf(
            [A.ColorJitter(p=1),
             A.ToGray(p=1)], p=0.3
        ),
        A.SmallestMaxSize(max_size=scales), ],
        bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3))
    return trans


def teacher_no_transforms_albu(res):
    trans = A.Compose([
        A.SmallestMaxSize(max_size=[res]), ], bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3))
    return trans


def teacher_transforms_plus_mm(res, isminiou=True):
    colortransformlist = [
        Tmm.Identity(),
        Tmm.AutoContrast(),
        Tmm.RandEqualize(),
        Tmm.RandColor(),
        Tmm.RandContrast(),
        Tmm.RandBrightness(),
        Tmm.RandSharpness(),
        Tmm.RandPosterize(),
    ]

    geotransform = [
        Tmm.RandTranslate(x=(-0.1, 0.1)),
        Tmm.RandTranslate(y=(-0.1, 0.1)),
        Tmm.RandRotate(angle=(-30, 30)),
        [
            Tmm.RandShear(x=(-30, 30)),
            Tmm.RandShear(y=(-30, 30)),
        ],
    ]
    resize = Tmm.MinIoURandomCrop() if isminiou else Tmm.RandResize(
        img_scale=[(1333, 400), (1333, 1200)], multiscale_mode="range", keep_ratio=True)
    trans = Tmm.BaseCompose([
        # Tmm.RandomCrop(crop_type='relative_range', crop_size=(0.2,0.2),bbox_clip_border=True),
        resize,
        Tmm.RandomFlip(flip_ratio=0.5),
        Tmm.Sequential(
            transforms=[
                Tmm.ShuffledSequential(
                    transforms=[
                        Tmm.OneOf(transforms=colortransformlist),
                        Tmm.OneOf(transforms=geotransform)
                    ]
                ),
                Tmm.RandErase(
                    n_iterations=(1, 5),
                    size=[0, 0.2],
                    squared=True)
            ]
        )
    ])
    return trans


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class BoxSafeCrop(A.RandomSizedBBoxSafeCrop):
    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = random_crop(img, crop_height, crop_width, h_start, w_start)
        return crop

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        # less likely, this class is for use with bboxes.
        if len(params["bboxes"]) == 0:
            erosive_h = int(img_h * (1.0 - self.erosion_rate))
            crop_height = img_h if erosive_h >= img_h else random.randint(
                erosive_h, img_h)
            return {
                "h_start": random.random(),
                "w_start": random.random(),
                "crop_height": crop_height,
                "crop_width": int(crop_height * img_w / img_h),
            }
        # get union of all bboxes
        x, y, x2, y2 = random.choice(params["bboxes"])[:4]
        # print('org box', [x, y, x2, y2])
        sw = x2-x
        sh = y2-x
        if sw >= sh:
            sh = sw
            bx1 = x
            y1_lower_delta = max(y+sh-1, 0)
            y1_upper_delta = max(sh-y2, 0)
            y1_lower = max(y-y1_lower_delta, 0)
            y1_upper = min(y+y1_upper_delta, 1)
            by1 = random.uniform(y1_lower, y1_upper)
            bx2 = bx1+sh
            by2 = by1+sh

        else:
            sw = sh
            by1 = y
            x1_lower_delta = max(x+sh-1, 0)
            x1_upper_delta = max(sh-x2, 0)
            x1_lower = max(x-x1_lower_delta, 0)
            x1_upper = min(x+x1_upper_delta, 1)
            bx1 = random.uniform(x1_lower, x1_upper)
#             bx1=random.uniform(max(0,x2-sw),min(1,x2+sw))
            bx2 = bx1+sh
            by2 = by1+sh
        # print('med box', [bx1, by1, bx2, by2])
        if bx1 > by1:
            d1 = random.random()*by1

        else:
            d1 = random.random()*bx1
        bx1 = bx1-d1
        by1 = by1-d1
        if (1-bx2) > (1-by2):
            d2 = random.random()*(1-by2)
        else:
            d2 = random.random()*(1-bx2)
        bx2 = bx2+d2
        by2 = by2+d2

        bx = bx1
        by = by1
        # print('crop region:', [bx, by, bx2, by2])

        bw, bh = bx2 - bx, by2 - by
        crop_height = img_h if bh >= 1.0 else int(img_h * bh)
        crop_width = img_w if bw >= 1.0 else int(img_w * bw)
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
        return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}
