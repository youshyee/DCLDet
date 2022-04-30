# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from .backbone import build_backbone
from .deformable_detr import DeformableDETR, SetCriterion as DefSetCriterion, PostProcess as DefPostProcess
from .detr import DETR, SetCriterion as DETRSetCriterion, PostProcess as DETRPostProcess
from .def_matcher import build_matcher as build_def_matcher
from .detr_matcher import build_matcher as build_detr_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer

from .transformer import build_transformer


def build_model(args, only_model=False):
    num_classes = 80
    device = torch.device(args.device)

    weight_dict = {'ce': args.cls_loss_coef,
                   'bbox': args.bbox_loss_coef,
                   'giou': args.giou_loss_coef
                   }
    losses = ['labels', 'boxes', 'cardinality']

    backbone = build_backbone(args)

    if args.model == 'deformable_detr':
        transformer = build_deforamble_transformer(args)
        model = DeformableDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            num_feature_levels=args.num_feature_levels,
            aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            two_stage=args.two_stage,
        )
        if only_model:
            return model
        matcher = build_def_matcher(args)
        criterion = DefSetCriterion(
            num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
        postprocessors = {'bbox': DefPostProcess()}

    elif args.model == 'detr':
        transformer = build_transformer(args)
        model = DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            object_embedding_loss=args.object_embedding_loss,
            obj_embedding_head=args.obj_embedding_head
        )
        if only_model:
            return model
        matcher = build_detr_matcher(args)
        criterion = DETRSetCriterion(num_classes, matcher, weight_dict, args.eos_coef,
                                     losses, object_embedding_loss=args.object_embedding_loss)
        postprocessors = {'bbox': DETRPostProcess()}
    else:
        raise ValueError("Wrong model.")

    criterion.to(device)

    return model, criterion, postprocessors
