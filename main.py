import argparse
import datetime
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import CocoDetection
import datasets.samplers as samplers
from datasets.trans import (
    make_self_det_transforms,
    teacher_no_transforms_albu,
    teacher_transforms_albu,
    teacher_transforms_plus_mm,
)
from engine import evaluate, get_statistic, train_one_epoch, viz
from models import build_model
import util.misc as utils
from util.parse_args import get_parser, handle_defaults


def main(args, **kwargs):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("{}".format(args).replace(', ', ',\n'))
    # utils.log_preprocess(args, **kwargs)
    device = torch.device(args.device)
    ##########
    #  seed  #
    ##########
    if args.random_seed:
        args.seed = np.random.randint(0, 1000000)
    if args.resume:
        checkpoint_args = torch.load(args.resume, map_location='cpu')['args']
        args.seed = checkpoint_args.seed
        print("Loaded random seed from checkpoint:", checkpoint_args.seed)
    elif (Path(args.output_dir)/'checkpoint.pth').exists():
        checkpoint_args = torch.load(
            str(Path(args.output_dir)/'checkpoint.pth'), map_location='cpu')['args']
        args.seed = checkpoint_args.seed
        print("Loaded random seed from checkpoint:", checkpoint_args.seed)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Using random seed: {seed}")
    ###########
    #  model  #
    ###########
    student, criterion, postprocessors = build_model(args)
    student.to(device)
    student = torch.nn.parallel.DistributedDataParallel(
        student, device_ids=[args.gpu])
    student_without_ddp = student.module
    teacher = build_model(args, only_model=True)
    teacher.to(device)
    n_parameters = sum(p.numel()
                       for p in student.parameters() if p.requires_grad)
    print('Number of params for Network: {}M'.format(n_parameters//1000000))
    #############
    #  dataset  #
    #############
    dataset_train, dataset_evals = get_datasets(args)

    if args.eval:
        dataset_val = dataset_evals[0]
        val_ann_file = os.path.join(args.data_ann, '../instances_val2017.json')
        dataset_val.reinit(0)
        print(dataset_val)
        sampler_val = samplers.DistributedSampler(
            dataset_val, shuffle=False)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                     pin_memory=True)
    if args.get_static:
        # TODO:  <28-03-22, Xinyu Yang> #
        pass
        list_annos = None
    # aug
    if args.no_aug:
        t_trans = teacher_no_transforms_albu(args.data_res)
        augmm = False
    else:
        if args.augplus:
            t_trans = teacher_transforms_plus_mm(args.data_res, args.miniou)
            augmm = True
        else:
            t_trans = teacher_transforms_albu(args.data_res)
            augmm = False
    ###############
    #  optimizer  #
    ###############
    param_dicts = dealwithparams(args, student_without_ddp)
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    output_dir = Path(args.output_dir)
    #################
    #  init models  #
    #################
    if args.pretrain:
        print('Initialized from the pre-training model')
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'student' in checkpoint:
            state_dict = checkpoint['student']
        else:
            state_dict = checkpoint
        msg = student_without_ddp.load_state_dict(
            state_dict, strict=False)
        print(msg)
    # init teacher
    teacher.load_state_dict(student_without_ddp.state_dict())
    # resume
    isresume = False
    resume = None
    if args.resume:
        isresume = True
        resume = args.resume
        checkpoint = torch.load(args.resume, map_location='cpu')
    elif (output_dir/'checkpoint.pth').exists():  # auto resume
        isresume = True
        resume = str(output_dir/'checkpoint.pth')
    if isresume:
        print("Resume from checkpoint: {}".format(resume))
        checkpoint = torch.load(resume, map_location='cpu')
        missing_keys, unexpected_keys = student_without_ddp.load_state_dict(
            checkpoint['student'], strict=True)
        outload = teacher.load_state_dict(checkpoint['teacher'], strict=True)
        print('loading teacher with output of ', outload)
        unexpected_keys = [k for k in unexpected_keys if not (
            k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    # viz
    if args.viz:
        viz(teacher, postprocessors, data_loader_test, device, args.output_dir)
        return
    momentum_schedule, unsup_schedule, filter_schedule, top_schedule, data_schedule = dealwithpolicies(
        args)

    ##############
    #  training  #
    ##############
    if args.finalval:
        args.start_epoch = args.epochs
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        reinitvalue = data_schedule[epoch]
        dataset_train.reinit(reinitvalue)
        print(dataset_train)
        sampler_train = samplers.DistributedGroupSampler(
            dataset_train, args.batch_size)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)  # keep the same batch size ==1 we will deal with the batch size in the loop
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn_no_nest, num_workers=args.num_workers,
                                       pin_memory=True)

        momentum_teacher = momentum_schedule[epoch]
        filter_value = filter_schedule[epoch]
        top_value = top_schedule[epoch]
        unsupweight = unsup_schedule[epoch]
        train_stats = train_one_epoch(
            teacher,
            student,
            criterion,
            postprocessors,
            data_loader_train,
            optimizer,
            t_trans,
            momentum_teacher,
            filter_value,
            top_value,
            device,
            epoch,
            args.clip_max_norm,
            batch_size=args.batch_size,
            scale_value=unsupweight,
            augmm=augmm,
        )
        lr_scheduler.step()

        # ################
        # #  checkpoint  #
        # ################
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch+1) > args.warmup_epochs and (epoch + 1) % args.save_every == 0:
                checkpoint_paths.append(
                    output_dir / f'checkpoint{epoch:04}.pth')
            elif (epoch+1) <= args.warmup_epochs and (epoch + 1) % int(1000/args.percent) == 0:
                checkpoint_paths.append(
                    output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'student': student.module.state_dict(),
                    'teacher': teacher.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if args.get_static:
            if (epoch + 1) % args.static_every == 0 or (epoch + 1) in [1, 20, 50]:
                print('geting statistic')
                if list_annos is None:
                    print('statistic GT annotation is None, loading..')
                    list_annos = [state_dataset.anno_by_index(
                        i) for i in range(len(state_dataset))]
                    print('statistic GT annotation loading completed')

                get_statistic(teacher, postprocessors, state_loader,
                              list_annos, epoch, device, args.output_dir)
        ################
        #  evaluation  #
        ################
        test_stats = {}
        if args.eval:
            if (epoch+1) > args.warmup_epochs and (epoch+1) % args.eval_every == 0 and (epoch+1) != args.epochs:
                print('evaluating')
                test_stats = evaluate(
                    teacher,
                    criterion,
                    postprocessors,
                    data_loader_val,
                    val_ann_file,
                    device,
                    args.output_dir,
                    dataset_val.label2cat
                )
                print(test_stats)
            elif (epoch+1) <= args.warmup_epochs and (epoch+1) % int(50/args.percent) == 0 and (epoch+1) != args.epochs:
                print('evaluating')
                test_stats = evaluate(
                    teacher,
                    criterion,
                    postprocessors,
                    data_loader_val,
                    val_ann_file,
                    device,
                    args.output_dir,
                    dataset_val.label2cat
                )
                print(test_stats)
        log_stats = {'epoch': epoch,
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            # for evaluation logs

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    ###################
    #  final testing  #
    ###################

    if args.eval:
        print('Final testing')
        test_stats = evaluate(
            teacher,
            criterion,
            postprocessors,
            data_loader_val,
            val_ann_file,
            device,
            args.output_dir,
            dataset_val.label2cat
        )
        log_stats = {
            **{f'final_test_{k}': v for k, v in test_stats.items()}
        }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        # utils.log_postprocess(args, mAP=test_stats['bbox_mAP'])


def get_datasets(args):
    print('Image path:', args.data_root_train)
    if args.percent != 100:
        sup_ann = os.path.join(
            args.data_ann, f'instances_train2017.{args.fold}@{args.percent}.json')
        unsup_ann = os.path.join(
            args.data_ann, f'instances_train2017.{args.fold}@{args.percent}-unlabeled.json')
    else:
        sup_ann = os.path.join(
            args.data_ann, f'../instances_train2017.json')
        unsup_ann = None
    print('unsupervised annotation path', unsup_ann)
    print('supervised annotation path', sup_ann)

    train_transforms = make_self_det_transforms(
        'train', args.data_res, isjitter=args.isjitter)
    test_transforms = make_self_det_transforms('val', args.data_res)
    teacher_transforms = make_self_det_transforms('teacher', args.data_res)

    dataset_train = CocoDetection(args.data_root_train,
                                  sup_ann,
                                  unsup_ann,
                                  train_transforms,
                                  teacher_transforms)
    dataset_evals = []
    if args.eval:
        dataset_evals.append(
            CocoDetection(args.data_root_val,
                          os.path.join(args.data_ann,
                                       '../instances_val2017.json'),
                          None,
                          test_transforms,
                          None,
                          val=True)
        )

    if args.get_static:
        # TODO:  <28-03-22, Xinyu Yang> #
        pass
    return dataset_train, dataset_evals


def dealwithpolicies(args):
    # curiculum learning strategies
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_start, args.momentum_end, args.epochs, 1, warmup_epochs=args.warmup_epochs
    )
    # unsupervised weight schedule linear increase from start to end
    unsup_schedule_mid = np.linspace(
        args.unsup_weight_start, args.unsup_weight_end, args.epochs-args.warmup_epochs-args.post_epochs)
    unsup_schedule_head = np.ones(args.warmup_epochs)*args.unsup_weight_start
    unsup_schedule_tail = np.ones(args.post_epochs)*args.unsup_weight_end
    unsup_schedule = np.concatenate(
        (unsup_schedule_head, unsup_schedule_mid, unsup_schedule_tail))
    assert len(unsup_schedule) == args.epochs

    # deal with threshold
    arctanspeed = 15
    if args.filter_arctan:
        # as arctan is flat in the end thus we would not consider cooldown phase
        def getx(start, end, speed=arctanspeed):
            return np.tan((start/end-0.05)*np.pi/2)/speed*1000
        c = np.arange(args.warmup_epochs, args.epochs)
        shiftx = getx(args.filter_start, args.filter_end)
        arctanpart = args.filter_end * \
            (np.arctan((c+shiftx-args.warmup_epochs)*arctanspeed/1000)/np.pi*2+0.05)
        filter_schedule = np.concatenate(
            (np.ones(args.warmup_epochs)*args.filter_start, arctanpart))
    else:
        filter_schedule_mid = np.linspace(
            args.filter_start, args.filter_end, args.epochs-args.warmup_epochs-args.post_epochs)
        filter_schedule_head = np.ones(args.warmup_epochs)*args.filter_start
        filter_schedule_tail = np.ones(args.post_epochs)*args.filter_end
        filter_schedule = np.concatenate(
            (filter_schedule_head, filter_schedule_mid, filter_schedule_tail))
    assert len(filter_schedule) == args.epochs

    # deal with data strategy
    # increace + warmup and cooldown
    up = np.concatenate((
        np.zeros(args.warmup_epochs),
        np.linspace(0, 1, args.epochs - args.warmup_epochs-args.post_epochs),
        np.ones(args.post_epochs)
    ))

    # decrease + warmup and cooldown
    down = np.concatenate((
        np.ones(args.warmup_epochs),
        np.linspace(1, 0, args.epochs - args.warmup_epochs-args.post_epochs),
        np.zeros(args.post_epochs)
    ))
    uplinear = np.linspace(0, 1, args.epochs)
    downlinear = np.linspace(1, 0, args.epochs)
    fix = np.linspace(1, 1, args.epochs)
    # set increase function

    if args.data_stragety == 'up':
        data_schedule = up
    elif args.data_stragety == 'down':
        data_schedule = down
    elif args.data_stragety == 'uplinear':
        assert args.no_sup_warmup, 'uplinear strategy only support no_sup_warmup'
        assert args.no_sup_cooldown, 'uplinear strategy only support no_sup_cooldown'
        data_schedule = uplinear
    elif args.data_stragety == 'downlinear':
        assert args.no_sup_warmup, 'downlinear strategy only support no_sup_warmup'
        assert args.no_sup_cooldown, 'downlinear strategy only support no_sup_cooldown'
        data_schedule = downlinear
    elif args.data_stragety == 'fix':
        assert args.no_sup_warmup, 'downlinear strategy only support no_sup_warmup'
        assert args.no_sup_cooldown, 'dowlinear strategy only support no_sup_cooldown'
        data_schedule = fix
    else:
        raise ValueError('data_stragety not supported')
    if args.data_fix:
        assert args.data_stragety == 'fix'
        assert args.no_sup_warmup, 'data_fix strategy only support no_sup_warmup'

    # deal with top
    top_schedule_mid = np.linspace(
        args.top_start, args.top_end, args.epochs-args.warmup_epochs-args.post_epochs)
    top_schedule_head = np.ones(args.warmup_epochs)*args.top_start
    top_schedule_tail = np.ones(args.post_epochs)*args.top_end
    top_schedule = np.concatenate(
        (top_schedule_head, top_schedule_mid, top_schedule_tail))
    assert len(top_schedule) == args.epochs
    return momentum_schedule, unsup_schedule, filter_schedule, top_schedule, data_schedule


def dealwithparams(args, student):
    if args.lr_backbone == 0:
        print("lr_backbone is 0, so no backbone parameters will be updated")
        param_dicts = [
            {
                "params":
                    [p for n, p in student.named_parameters()
                     if not utils.match_name_keywords(n, args.lr_backbone_names) and not utils.match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in student.named_parameters() if utils.match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
            }
        ]
    else:
        param_dicts = [
            {
                "params":
                    [p for n, p in student.named_parameters()
                     if not utils.match_name_keywords(n, args.lr_backbone_names) and not utils.match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in student.named_parameters() if utils.match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": float(args.lr_backbone),
            },

            {
                "params": [p for n, p in student.named_parameters() if utils.match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
            }
        ]
    print('initialise the model with lr:' + str(args.lr), 'lr_linear_proj_mult:' + str(args.lr_linear_proj_mult),
          'lr_backbone:' + str(args.lr_backbone))
    return param_dicts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'dynamic policies for COCO semi-supervised object detection training and evaluation script', parents=[get_parser()])
    parser.add_argument('--finalval', action="store_true")
    args = parser.parse_args()
    args, changed = handle_defaults(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # args assertation
    if args.no_sup_warmup:
        assert args.warmup_epochs == 0
    if args.no_sup_cooldown:
        assert args.post_epochs == 0
    if args.lr_backbone is None:
        args.lr_backbone = args.lr * 0.1
    main(args, changed=changed)
