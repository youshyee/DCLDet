# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os
from pathlib import Path

from PIL import Image
import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import torch
import torch.utils.data
from torch.utils.data import Dataset
from terminaltables import AsciiTable

import datasets.transforms as T
import random



class CocoDetection(Dataset):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self,
                 img_folder,
                 ann_file_sup,
                 ann_file_unsup,  # None for only sup input
                 transforms_student,
                 transforms_teacher,
                 no_cats=False,
                 use_mask=False,
                 val=False,
                 filter_empty_gt=True,
                 ):
        self.transforms_student = transforms_student
        self.transforms_teacher = transforms_teacher
        self.no_cats = no_cats
        self.img_folder = img_folder
        self.val = val
        self.use_mask = use_mask
        self.filter_empty_gt = filter_empty_gt
        self.coco_sup = COCO(ann_file_sup)
        self.only_sup = ann_file_unsup is None
        if not self.only_sup:
            self.coco_unsup = COCO(ann_file_unsup)
        self.cat_ids = self.coco_sup.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.label2cat = { i:cat_id for i, cat_id in enumerate(self.cat_ids)}
        if self.val:
            assert ann_file_unsup is None

        self.image_infos_sup_ = self.get_infos(
            self.coco_sup, issupervised=True)
        if not self.only_sup:
            self.image_infos_unsup_ = self.get_infos(
                self.coco_unsup, issupervised=False)
        self.filtered_len_sup = 0
        if not self.val:
            sup_len_before = len(self.image_infos_sup_)
            valid_idx_sup = self._filter_imgs_sup()
            self.image_infos_sup_ = [self.image_infos_sup_[i]
                                     for i in valid_idx_sup]
            self.filtered_len_sup = sup_len_before-len(self.image_infos_sup_)
            if not self.only_sup:
                self.filtered_len_unsup = 0
                unsup_len_before = len(self.image_infos_unsup_)
                valid_idx_unsup = self._filter_imgs_unsup()
                self.image_infos_unsup_ = [self.image_infos_unsup_[i]
                                           for i in valid_idx_unsup]
                self.filtered_len_unsup = unsup_len_before - \
                    len(self.image_infos_unsup_)

        self._set_group_flag()
        if not self.only_sup:
            self.unsup_len = len(self.image_infos_unsup_)
            np.random.shuffle(self.image_infos_unsup_)
        self.image_infos = []

    def reinit(self, ratio):
        assert ratio >= 0 and ratio <= 1
        self.image_infos = [] + self.image_infos_sup_
        self.flag = self.flag_sup.copy()

        if not self.only_sup:
            sampled_image_infos = [] + \
                self.image_infos_unsup_[:int(self.unsup_len * ratio)]
            self.image_infos += sampled_image_infos
            self.flag = np.concatenate(
                [self.flag, self.flag_unsup[:int(self.unsup_len * ratio)]]
            )
            print('Reinitializing dataset with ratio {}, training on sup {} images and unsup {} images.'.format(
                ratio, len(self.image_infos_sup_), len(sampled_image_infos)))
        assert len(self.image_infos) == len(self.flag)

    def get_infos(self, coco, issupervised):
        img_ids = coco.getImgIds()
        img_infos = []
        total_ann_ids = []
        for i in img_ids:
            meta = coco.loadImgs([i])[0]
            meta['filename'] = meta['file_name']
            meta['issupervised'] = issupervised
            img_infos.append(meta)
            ann_ids = coco.getAnnIds(imgIds=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids  are not unique!"
        return img_infos

    def __getitem__(self, idx):
        img, target, issupervised = self.parse_info(
            idx, return_masks=self.use_mask)
        if issupervised:
            img, target = self.transforms_student(img, target)
        else:
            img, target = self.transforms_teacher(img, target)

        if issupervised and not self.val:
            if len(target['boxes']) < 1:
                return self.__getitem__(random.randint(0, len(self) - 1))
        if self.no_cats:
            target['labels'][:] = 1
        return img, target

    def _filter_imgs_sup(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco_sup.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco_sup.catToImgs[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        for i, img_info in enumerate(self.image_infos_sup_):
            img_id = img_info['id']
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _filter_imgs_unsup(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id']
                           for _ in self.coco_unsup.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco_unsup.catToImgs[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        for i, img_info in enumerate(self.image_infos_unsup_):
            img_id = img_info['id']
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def __len__(self):
        return len(self.image_infos)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag_sup = np.zeros(len(self.image_infos_sup_), dtype=np.uint8)
        for i in range(len(self.image_infos_sup_)):
            img_info = self.image_infos_sup_[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag_sup[i] = 1
        if not self.only_sup:
            self.flag_unsup = np.zeros(
                len(self.image_infos_unsup_), dtype=np.uint8)
            for i in range(len(self.image_infos_unsup_)):
                img_info = self.image_infos_unsup_[i]
                if img_info['width'] / img_info['height'] > 1:
                    self.flag_unsup[i] = 1

    def parse_info(self, idx, return_masks):
        img_info = self.image_infos[idx]
        img_id = img_info['id']
        issupervised = img_info['issupervised']
        image = Image.open(os.path.join(self.img_folder, img_info['filename'])).convert('RGB')
        w, h = image.size
        # image = torch.tensor(np.array(image))
        if issupervised:
            target_ids = self.coco_sup.getAnnIds(imgIds=[img_id])
            targets = self.coco_sup.loadAnns(target_ids)

            targets = [
                obj for obj in targets if 'iscrowd' not in obj or obj['iscrowd'] == 0]

            boxes = [obj["bbox"] for obj in targets]
            # guard against no boxes via resizing
            boxes, keep = preprocess_xywh_boxes(boxes, h, w)

            classes = [self.cat2label[obj["category_id"]] for obj in targets]
            classes = torch.tensor(classes, dtype=torch.int64)
            classes = classes[keep]

            if return_masks:
                segmentations = [obj["segmentation"] for obj in targets]
                masks = convert_coco_poly_to_mask(segmentations, h, w)
                masks = masks[keep]

            keypoints = None
            if targets and "keypoints" in targets[0]:
                keypoints = [obj["keypoints"] for obj in targets]
                keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
                num_keypoints = keypoints.shape[0]
                if num_keypoints:
                    keypoints = keypoints.view(num_keypoints, -1, 3)
                keypoints = keypoints[keep]
            area = torch.tensor([obj["area"] for obj in targets])
            iscrowd = torch.tensor(
                [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in targets])
            area = area[keep]
            iscrowd = iscrowd[keep]
        ###################################
        #  unsupervised load dummy label  #
        ###################################
        else:
            boxes = torch.ones((1, 4))*-1
            classes = torch.tensor([-1], dtype=torch.int64)
            if return_masks:
                masks = torch.ones((1, h, w))*-1
            keypoints = None
            area = torch.tensor([0])
            iscrowd = torch.tensor([-1])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if return_masks:
            target["masks"] = masks
        target["image_id"] = torch.tensor(img_id)
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        target['issupervised'] = torch.tensor(issupervised)
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target, issupervised

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Val' if self.val else 'Train'
        unsupfilterlen = self.filtered_len_unsup if hasattr(
            self, 'filtered_len_unsup') else 0
        unsuplen = len(self.image_infos_unsup_) if hasattr(
            self, 'image_infos_unsup_') else 0
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                  f'with number of supervised images {len(self.image_infos_sup_)}, '
                  f'filter supervised images {self.filtered_len_sup}, '
                  f'with number of all imgs {len(self)}, '
                  f'and instance counts: \n')
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result

        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        # count the instance number in each image
        for img_info in self.image_infos:
            img_id = img_info['id']
            issupervised = img_info['issupervised']
            if issupervised:
                target_ids = self.coco_sup.getAnnIds(imgIds=[img_id])
                targets = self.coco_sup.loadAnns(target_ids)
            else:
                target_ids = self.coco_unsup.getAnnIds(imgIds=[img_id])
                targets = self.coco_unsup.loadAnns(target_ids)
            targets = [
                obj for obj in targets if 'iscrowd' not in obj or obj['iscrowd'] == 0]
            labels = [self.cat2label[obj["category_id"]] for obj in targets]
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) > 0:
                # add the occurrence number to each class
                instance_count[unique] += counts
            else:
                # background is the last index
                instance_count[-1] += 1
        # create a table with category count
        table_data = [['category', 'count'] * 5]
        row_data = []
        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f'{cls} [{self.CLASSES[cls]}]', f'{count}']
            else:
                # add the background number
                row_data += ['-1 background', f'{count}']
            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []
        if len(row_data) >= 2:
            if row_data[-1] == '0':
                row_data = row_data[:-2]
            if len(row_data) >= 2:
                table_data.append([])
                table_data.append(row_data)

        table = AsciiTable(table_data)
        result += table.table
        return result


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def preprocess_xywh_boxes(boxes, h, w):
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    return boxes, keep


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
