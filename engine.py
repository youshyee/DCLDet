# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import tqdm
import util.misc as utils
import util.helper as helper
from datasets.coco_eval import CocoEvaluator
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
import tqdm
from functools import partial
from pathlib import Path
from pycocotools.coco import COCO

import torchvision
from torchvision.transforms.functional import to_tensor


def train_one_epoch(teacher: torch.nn.Module,
                    student: torch.nn.Module,
                    criterion: torch.nn.Module,
                    postprocessors: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    t_trans,
                    momentum_teacher,
                    filter_value,
                    top_value,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0,
                    batch_size: int = 1,
                    view_output=False,
                    scale_value=0.5,
                    augmm=False,
                    ):
    student.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('prob_max_mean', utils.SmoothedValue(
        window_size=1000, fmt=None))
    metric_logger.add_meter('p3', utils.SmoothedValue(
        window_size=1000, fmt=None))
    metric_logger.add_meter('p5', utils.SmoothedValue(
        window_size=1000, fmt=None))
    metric_logger.add_meter('p7', utils.SmoothedValue(
        window_size=1000, fmt=None))
    metric_logger.add_meter('p9', utils.SmoothedValue(
        window_size=1000, fmt=None))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    mean_tensor_ = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_tensor_ = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    count = 0

    if augmm:
        transfunc = partial(helper.teacher_transforms_single_mm, trans=t_trans)
    else:
        transfunc = partial(helper.teacher_transforms_single, trans=t_trans)
    for out in metric_logger.log_every(data_loader, print_freq, header=header ):
        count+=1
        samples, targets= out # list of tensors and targets
        # split sup and unsuper
        with torch.no_grad():
            issupervised=torch.stack([t['issupervised'] for t in targets],dim=0)

            # get unsupervised
            unsupervised_tensor=[]
            for b in range(batch_size):
                if issupervised[b].item()==0:
                    unsupervised_tensor.append(samples[b])

            # handle unsupervised lable and aug
            if len(unsupervised_tensor)!=0: # this case use teacher model
                unsupervised_samples=utils.nested_tensor_from_tensor_list(unsupervised_tensor).to(device, non_blocking=True)
                pseduo_bboxes = teacher_forward(teacher, unsupervised_samples, postprocessors, filter_value=filter_value, top_value=top_value)
                num_bx = sum([len(b) for b in pseduo_bboxes])
                if num_bx == 0:
                    print("No bbox found with filter_value of {}".format(
                        filter_value))
                # BC,H,W channel last
                unsupervised_samples_masks=~unsupervised_samples.mask
                unsupervised_samples=helper.batchtensor2numpyimg(unsupervised_samples.tensors)
                num_unsuper=len(unsupervised_samples)
                out_samples = []
                out_boxes = []
                for i in range(num_unsuper):
                    w=unsupervised_samples_masks[i].sum(dim=0).bool().sum().item()
                    h=unsupervised_samples_masks[i].sum(dim=1).bool().sum().item()
                    out,save_org_box = transfunc(single_task=(unsupervised_samples[i][0], pseduo_bboxes[i],(w,h)))
                    if view_output:
                        save_org=unsupervised_samples[i][0]
                        save_img=out['image']
                        save_bboxes=out['bboxes']
                        torch.save((save_img,save_bboxes,save_org,save_org_box),f'./views/img_box_{count}_{i}.pt')
                    out_samples.append(to_tensor(out['image']).sub(mean_tensor_).div(std_tensor_))
                    if len(out['bboxes']) == 0:
                        box = torch.tensor([]).view(0,4)
                    else:
                        box = box_xyxy_to_cxcywh(
                            torch.tensor(out['bboxes'])[:, :4]).float()
                    out_boxes.append(box)

                # merge with the supvised
                student_samples=[]
                newtargets = []
                out_sample_index=0
                for i in range(batch_size):
                    newtarget = {k: v.to(device, non_blocking=True)
                                 for k, v in targets[i].items()}
                    if issupervised[i].item()==1:
                        student_samples.append(samples[i])
                    else:
                        student_samples.append(out_samples[out_sample_index])
                        bboxes_=out_boxes[out_sample_index]
                        newtarget['boxes'] = bboxes_.to(device, non_blocking=True)
                        newtarget['labels'] = torch.zeros(bboxes_.shape[0]).long().to(device)
                        out_sample_index+=1
                    newtargets.append(newtarget)
                assert out_sample_index==num_unsuper
                student_samples = utils.nested_tensor_from_tensor_list(
                    student_samples).to(device, non_blocking=True)
                targets = newtargets

            else:
                targets = [{k: v.to(device, non_blocking=True)
                            for k, v in t.items()} for t in targets]

                student_samples = utils.nested_tensor_from_tensor_list(
                    samples).to(device, non_blocking=True)

            sup_box_mask=[]
            for b in range(batch_size):
                if issupervised[b].item()==0:
                    sup_box_mask+=[False]*len(targets[b]['boxes'])
                else:
                    sup_box_mask+=[True]*len(targets[b]['boxes'])
            sup_box_mask=torch.tensor(sup_box_mask).bool().to(device)
            sup_batch_mask=issupervised.bool().to(device)
        outputs = student(student_samples)

        loss_dict = criterion(outputs, targets,None)
        weight_dict = criterion.weight_dict
        sup_dic,unsup_dic=helper.aggregate_loss(loss_dict,sup_box_mask,sup_batch_mask,device)

        sup_loss = sum(sup_dic[k] * weight_dict[k]
                     for k in sup_dic.keys())
        unsup_loss = sum(unsup_dic[k] * weight_dict[k]
                     for k in unsup_dic.keys())
        losses=sup_loss+unsup_loss*scale_value
        losses=losses/(1+scale_value)
        sup_dic.update({'loss_all': sup_loss})
        unsup_dic.update({'loss_all': unsup_loss})

        sup_loss_dict_reduced = utils.reduce_dict(sup_dic)
        unsup_loss_dict_reduced = utils.reduce_dict(unsup_dic)

        losses_reduced_scaled = utils.reduce_dict({'loss': losses,})
        loss_value = losses_reduced_scaled['loss'].item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(sup_loss_dict_reduced,unsup_loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(
                student.parameters(), max_norm)
        optimizer.step()

        # get statistics the confidence score
        out_logits = outputs['pred_logits']
        # [N, object queries num, 1:class]
        prob = out_logits.sigmoid().reshape(batch_size, -1)
        p3 = torch.sum(prob >= 0.3).item()/batch_size
        p5 = torch.sum(prob >= 0.5).item()/batch_size
        p7 = torch.sum(prob >= 0.7).item()/batch_size
        p9 = torch.sum(prob >= 0.9).item()/batch_size
        metric_logger.update(p3=p3)
        metric_logger.update(p5=p5)
        metric_logger.update(p7=p7)
        metric_logger.update(p9=p9)
        prob_max, _ = prob.max(dim=1)
        prob_mean = prob_max.mean().cpu().item()
        metric_logger.update(prob_max_mean=prob_mean)
        sup_loss_dict_reduced={'sup_'+k:v for k,v in sup_loss_dict_reduced.items()}
        unsup_loss_dict_reduced={'unsup_'+k:v for k,v in unsup_loss_dict_reduced.items()}
        metric_logger.update(loss=loss_value,**unsup_loss_dict_reduced,**sup_loss_dict_reduced)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(grad_norm=grad_total_norm)
        # metric_logger.update(momentum_teacher=momentum_teacher)
        # metric_logger.update(filter_value=filter_value)
        with torch.no_grad():
            m = momentum_teacher  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    lr_epoch=optimizer.param_groups[0]["lr"]
    lr_bk=optimizer.param_groups[1]["lr"]
    print("Averaged stats:", 'lr:',lr_epoch,  'lr_bk:',lr_bk,'momentum:',momentum_teacher, 'filter_value:',filter_value,'top_value',round(top_value), metric_logger)
    outmeters={k: meter.global_avg for k, meter in metric_logger.meters.items()}
    outmeters.update({'lr':lr_epoch,  'lr_bk':lr_bk,'momentum':momentum_teacher, 'filter_value':filter_value,'top_value':round(top_value), 'unsup_weight':scale_value})
    return outmeters


@torch.no_grad()
def teacher_forward(teacher, samples, postprocessors, filter_value=0.3, top_value=10):
    top_value = round(top_value)
    teacher.eval()
    bs = len(samples.tensors)  # nested tensor
    outputs = teacher(samples)
    # show results
    topk=int(teacher.num_queries/3)
    results = postprocessors['bbox'](
        outputs, torch.zeros(bs, 2), topk=topk, istargetsize=False)
    scores = [i['scores'] for i in results]
    predictied_boxes = [i['boxes'].clamp(0., 1.) for i in results]

    # filter prediction
    scores_filter = [s > filter_value for s in scores]
    predictied_boxes = [p[s] for p, s in zip(predictied_boxes, scores_filter)]

    # filter by min boxes
    min_box_filter = [(b[:, 2]*b[:, 3]) > 0.0015 for b in predictied_boxes]
    predictied_boxes = [p[s, :]
                        for p, s in zip(predictied_boxes, min_box_filter)]

    # add cls label for albulation process
    predictied_boxes = [torch.cat([
        p[:top_value, :].cpu(),
        torch.zeros(p[:top_value].shape[0], 1)
    ], dim=1).tolist() for p in predictied_boxes]

    return predictied_boxes


@torch.no_grad()
def evaluate(model,
             criterion,
             postprocessors,
             data_loader,
             val_ann_file,
             device,
             output_dir,
             label2cat,
             topk=100):
    model.eval()
    criterion.eval()
    iou_types = tuple(k for k in ('segm', 'bbox')
                      if k in postprocessors.keys())

    coco_evaluator = CocoEvaluator(COCO(val_ann_file), iou_types,label2cat)  # iou_types=bbox

    for samples, targets in tqdm.tqdm(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, topk=topk)

        save_val_results=False
        if save_val_results:
            Path(output_dir/'eval_bbox'/f'epoch{epoch}').mkdir(parents=True, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            for i, target in enumerate(targets):
                image_id = target['image_id'].item()
                pred_logits = outputs['pred_logits'][i]
                pred_boxes = outputs['pred_boxes'][i]
                img_h, img_w = target['orig_size']
                pred_boxes_ = box_cxcywh_to_xyxy(
                    pred_boxes) * torch.stack([img_w, img_h, img_w, img_h], dim=-1)
                torch.save(dict(image_id=image_id, target=target, pred_logits=pred_logits, pred_boxes=pred_boxes,
                                pred_boxes_=pred_boxes_), os.path.join(output_dir, str(image_id) + '.pt'))

        res = {target['image_id'].item(): output for target,output in zip(targets, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    mAP=coco_evaluator.summarize()
    stats=mAP
    return stats


@torch.no_grad()
def viz(model, postprocessors, data_loader, device, output_dir):
    ds=data_loader.dataset
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    out={}
    savepath=Path(output_dir)/f'viz_data'
    savepath.mkdir(exist_ok=True)
    local_rank=utils.get_rank()

    for samples, targets in tqdm.tqdm(data_loader):
        assert 'image_id' in targets[0]
        indexes=[t['image_id'].item() for t in targets]

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, topk=50)
        scores = [i['scores'].cpu() for i in results]
        predictied_boxes = [i['boxes'].cpu() for i in results]
        for s,b,i in zip(scores,predictied_boxes,indexes):
            idx=i
            frame_path=ds.targets[idx]
            out[idx]=(s,b,frame_path)
    torch.save(out,str(savepath/f'testset_{local_rank}.pth'))


@torch.no_grad()
def teacher_forward_static(teacher, samples,targets, postprocessors, filter_value=0.05,return_score=True,scale_to_orig=True):
    teacher.eval()
    bs = len(samples.tensors)  # nested tensor
    outputs = teacher(samples)
    target_sizes = torch.stack(
        [t["orig_size"] for t in targets], dim=0)
    results = postprocessors['bbox'](
        outputs, target_sizes, topk=100, istargetsize=True)
    scores = [i['scores'] for i in results]
    predictied_boxes = [i['boxes'] for i in results]

    scores_filter = [s > filter_value for s in scores]
    scores = [p[s].cpu() for p, s in zip(scores, scores_filter)]
    predictied_boxes = [p[s].cpu() for p, s in zip(predictied_boxes, scores_filter)]
    return predictied_boxes,scores
@torch.no_grad()
def get_statistic(model, postprocessors, data_loader,list_annos, epoch, device, output_dir):
    '''
    list_annos list of all annos with (n,4) tensor
    '''
    savepath=Path(output_dir)/f'epoch_{epoch}'
    savepath.mkdir(exist_ok=True)
    local_rank=utils.get_rank()
    model.eval()
    out={}
    for samples, targets in tqdm.tqdm(data_loader):
        assert 'image_id' in targets[0]
        indexes=[t['image_id'].item() for t in targets]
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        predictied_boxes,scores= teacher_forward_static(model, samples, targets, postprocessors, filter_value=0.1) # list of tensor
        for i in range(len(predictied_boxes)):
            imgidx=indexes[i]
            gt_boxes=list_annos[imgidx]
            score=scores[i]
            psudeo_boxes=predictied_boxes[i]
            # assert gt and predictied_boxes
            assert all(gt_boxes[:,0]<=gt_boxes[:,2]) and all(gt_boxes[:,1]<=gt_boxes[:,3])
            assert all(psudeo_boxes[:,0]<=psudeo_boxes[:,2]) and all(psudeo_boxes[:,1]<=psudeo_boxes[:,3])

            iou_matrix = torchvision.ops.box_iou(gt_boxes, psudeo_boxes)
            if iou_matrix.numel()==0:
                pass
                max_ious=torch.zeros(len(psudeo_boxes))
                gt_ious=torch.zeros(len(gt_boxes))
            else:
                max_ious = iou_matrix.max(dim=0)[0]
                gt_ious = iou_matrix.max(dim=1)[0] # for recall analysis
            assert len(score)==len(max_ious)
            assert len(gt_boxes)==len(gt_ious)
            out[imgidx]=(max_ious.cpu(),score.cpu(),gt_ious.cpu(),iou_matrix.cpu())
    torch.save(out,str(savepath/f'statistic_{local_rank}.pth'))


