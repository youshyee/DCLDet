import argparse
import numpy as np


default_args = dict(
    model='deformable_detr',
    fold=-1,
    percent=-1,
    ############
    #  policy  #
    ############
    unsup_weight_start=0.1,
    unsup_weight_end=1.0,
    filter_start=0.3,
    filter_end=0.5,
    filter_arctan=False,
    top_start=10,
    top_end=30,
    momentum_start=0.998,
    momentum_end=0.9998,
    no_sup_warmup=False,
    no_sup_cooldown=False,
    batch_size=6,
    warmup_epochs=40,
    post_epochs=20,
    ##############
    #  training  #
    ##############
    lr=0.0002,
    epochs=120,
    lr_drop=110,
    seed=2049,
    eval=True,
    viz=False,
    resume='',
    pretrain='',
    random_seed=False,
    eval_every=3,
    save_every=20,
    static_every=10,
    start_epoch=0,
    lr_backbone=None,
    lr_backbone_names=['backbone.0'],
    lr_linear_proj_names=['reference_points', 'sampling_offsets'],
    lr_linear_proj_mult=0.1,
    weight_decay=0.0001,
    lr_drop_epochs=None,
    clip_max_norm=0.1,
    sgd=False,
    final_test=True,
    output_dir='',
    ##########
    #  data  #
    ##########
    augplus=False,
    miniou=True,
    no_aug=False,
    isjitter=False,
    data_res=800,
    data_ann='data/coco/annotations/semi_supervised',  # annotation dir
    data_root_train='data/coco/train2017',
    data_root_val='data/coco/val2017',
    data_fix=False,
    data_stragety='up',
    ###########
    #  model  #
    ###########
    num_queries=300,
    filter_num=-1,
    reset_embedding_layer=1,
    with_box_refine=False,
    two_stage=False,
    obj_embedding_head='intermediate',
    backbone='resnet50',
    dilation=False,
    position_embedding='sine',
    position_embedding_scale=6.283185307179586,
    num_feature_levels=4,
    enc_layers=6,
    dec_layers=6,
    dim_feedforward=1024,
    hidden_dim=256,
    dropout=0.1,
    nheads=8,
    dec_n_points=4,
    enc_n_points=4,
    load_backbone='swav',
    masks=False,
    aux_loss=True,
    set_cost_class=2,
    set_cost_bbox=5,
    set_cost_giou=2,
    object_embedding_coef=1,
    mask_loss_coef=1,
    dice_loss_coef=1,
    cls_loss_coef=2,
    bbox_loss_coef=5,
    giou_loss_coef=2,
    eos_coef=0.1,
    focal_alpha=0.25,
    get_static=False,
    coco_panoptic_path=None,
    remove_difficult=False,
    cache_path=None,
    ############
    #  others  #
    ############
    device='cuda',
    num_workers=5,
    cache_mode=False,
    pre_norm=False
)


def get_parser():
    parser = argparse.ArgumentParser(
        'Deformable DETR Detector', add_help=False)
    parser.add_argument('name', type=str)
    parser.add_argument('percent', type=int)
    parser.add_argument('fold', type=int)
    parser.add_argument('--model', type=str)
    parser.add_argument('--unsup_weight_start', type=float)
    parser.add_argument('--unsup_weight_end', type=float)
    parser.add_argument('--filter_start', type=float)
    parser.add_argument('--filter_end', type=float)
    parser.add_argument('--filter_arctan', action='store_const', const=True)
    parser.add_argument('--top_start', type=int)
    parser.add_argument('--top_end', type=int)
    parser.add_argument('--momentum_start', type=float)
    parser.add_argument('--momentum_end', type=float)
    parser.add_argument('--no_sup_warmup', action='store_const', const=True)
    parser.add_argument('--no_sup_cooldown', action='store_const', const=True)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--warmup_epochs', type=int)
    parser.add_argument('--post_epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr_drop', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--eval', action='store_const', const=True)
    parser.add_argument('--viz', action='store_const', const=True)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrain', type=str)
    parser.add_argument('--random_seed', action='store_const', const=True)
    parser.add_argument('--eval_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--get_static', action='store_const', const=True)
    parser.add_argument('--static_every', type=int)
    parser.add_argument('--start_epoch', type=int)
    parser.add_argument('--lr_backbone', type=float)
    parser.add_argument('--lr_backbone_names', type=str)
    parser.add_argument('--lr_linear_proj_names', type=list)
    parser.add_argument('--lr_linear_proj_mult', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--lr_drop_epochs', type=int)
    parser.add_argument('--clip_max_norm', type=float)
    parser.add_argument('--sgd', action='store_const', const=True)
    parser.add_argument('--final_test', action='store_const', const=True)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--augplus', action='store_const', const=True)
    parser.add_argument('--miniou', type=bool)
    parser.add_argument('--isjitter', action='store_const', const=True)
    parser.add_argument('--no_aug', action='store_const', const=True)
    parser.add_argument('--data_ann', type=str)
    parser.add_argument('--data_res', type=int)
    parser.add_argument('--data_root_train', type=str)
    parser.add_argument('--data_root_val', type=str)
    parser.add_argument('--data_fix', action='store_const', const=True)
    parser.add_argument('--data_stragety', type=str)
    parser.add_argument('--num_queries', type=int)
    parser.add_argument('--filter_num', type=int)
    parser.add_argument('--reset_embedding_layer', type=int)
    parser.add_argument('--with_box_refine', type=bool)
    parser.add_argument('--two_stage', type=bool)
    parser.add_argument('--obj_embedding_head', type=str)
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--dilation', type=bool)
    parser.add_argument('--position_embedding', type=str)
    parser.add_argument('--position_embedding_scale', type=float)
    parser.add_argument('--num_feature_levels', type=int)
    parser.add_argument('--enc_layers', type=int)
    parser.add_argument('--dec_layers', type=int)
    parser.add_argument('--dim_feedforward', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--nheads', type=int)
    parser.add_argument('--dec_n_points', type=int)
    parser.add_argument('--enc_n_points', type=int)
    parser.add_argument('--load_backbone', type=str)
    parser.add_argument('--masks', type=bool)
    parser.add_argument('--aux_loss', type=bool)
    parser.add_argument('--set_cost_class', type=float)
    parser.add_argument('--set_cost_bbox', type=float)
    parser.add_argument('--set_cost_giou', type=float)
    parser.add_argument('--object_embedding_coef', type=float)
    parser.add_argument('--mask_loss_coef', type=float)
    parser.add_argument('--dice_loss_coef', type=float)
    parser.add_argument('--cls_loss_coef', type=float)
    parser.add_argument('--bbox_loss_coef', type=float)
    parser.add_argument('--giou_loss_coef', type=float)
    parser.add_argument('--eos_coef', type=float)
    parser.add_argument('--focal_alpha', type=float)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', type=bool)
    parser.add_argument('--cache_path', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--cache_mode', type=bool)
    parser.add_argument('--pre_norm', type=bool)
    return parser


def handle_defaults(args, default_args=default_args):
    changed = {}
    runtime_args = vars(args)
    for k, v in default_args.items():
        args_v = runtime_args[k]
        if v is not None:
            assert type(args_v) == type(
                v) or args_v is None, f'{k} is {type(v)} not of type {type(args_v)}'
        if args_v is None:
            setattr(args, k, v)
        else:
            changed[k] = args_v
    return args, changed
