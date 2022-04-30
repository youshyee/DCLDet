import torch
import numpy as np
from util.box_ops import box_cxcywh_to_xyxy_list, box_cxcywh_to_xyxy
from util.misc import get_world_size
import mmcv


mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)


# @torch.jit.script
def batchtensor2numpyimg(batch):
    """
    Convert a batch of tensor to numpy image
    :param batch: a batch of tensor
    :return: list of numpy image
    """
    batch = batch.permute(
        0, 2, 3, 1).cpu()
    batchlen = batch.shape[0]
    # unnormal
    batch.mul_(std_tensor).add_(mean_tensor)
    batch = batch*255
    batch = batch.clamp(0, 255)
    batch = batch.byte().numpy()
    batch = np.split(batch, batchlen)
    return batch


def aggregate_loss(loss_dict, sup_box_mask, sup_batch_mask, device, box_key=['giou', 'bbox'], batch_key=['ce']):
    allkey = box_key+batch_key
    loss = {k: [] for k in allkey}
    for key in loss:
        for k, v in loss_dict.items():
            if key in k:
                loss[key].append(v)

    sup_boxes = torch.sum(sup_box_mask)
    # sup_boxes = torch.as_tensor(
    #     [sup_boxes], dtype=torch.float, device=device)

    unsup_boxes = torch.sum(~sup_box_mask)
    # unsup_boxes = torch.as_tensor(
    #     [unsup_boxes], dtype=torch.float, device=device)

    reduce_stack = torch.stack([unsup_boxes, sup_boxes])
    torch.distributed.all_reduce(reduce_stack)
    unsup_boxes = reduce_stack[0]
    sup_boxes = reduce_stack[1]
    unsup_boxes = torch.clamp(
        unsup_boxes / get_world_size(), min=1e-2).item()
    sup_boxes = torch.clamp(
        sup_boxes / get_world_size(), min=1e-2).item()
    # print('sup_boxes: ',sup_boxes,'unsup_boxes: ',unsup_boxes)

    loss = {k: sum(v) for k, v in loss.items()}
    loss_sup = {k: [] for k in allkey}
    loss_unsup = {k: [] for k in allkey}
    for k, v in loss.items():
        if k in box_key:
            loss_sup[k] = v[sup_box_mask].sum()/sup_boxes
            loss_unsup[k] = v[~sup_box_mask].sum()/unsup_boxes
        elif k in batch_key:
            loss_sup[k] = v[sup_batch_mask].sum()/sup_boxes
            loss_unsup[k] = v[~sup_batch_mask].sum()/unsup_boxes
    return loss_sup, loss_unsup


def teacher_transforms_single(trans, single_task):
    '''
    single_task: [0] img numpy array
                [1] boxes list of list
                [2]orig_size (w,h)
    '''
    img = single_task[0]
    boxes = single_task[1]
    orig_size = single_task[2]
    boxes = [box_cxcywh_to_xyxy_list(b) for b in boxes]
    out = trans(image=img, bboxes=boxes)
    return out, boxes


def teacher_transforms_single_mm(trans, single_task):
    '''
    single_task: [0] img numpy array
                [1] boxes list of list
                [2]orig_size (w,h)
    return transformed img (np) and box (np or list)
    '''
    img = single_task[0]
    boxes = single_task[1]
    w, h = single_task[2]
    boxes = [box_cxcywh_to_xyxy_list(b) for b in boxes]
    boxes = np.array(boxes).reshape(-1, 5)
    boxes = boxes[:, :4]
    scale = np.array([w, h, w, h]).reshape(1, 4)

    results = {'img_fields': ['img'],
               'bbox_fields': ['boxes'],
               'img_shape': img.shape,
               'img': img,
               'boxes': boxes*scale,
               #          'gt_labels':[0,0],
               }
    results = trans(results)
    h, w, _ = results['img_shape']
    scale_back = np.array([w, h, w, h]).reshape(1, 4)
    out = {
        'image': results['img'],
        'bboxes': results['boxes']/scale_back,
    }
    return out, boxes
