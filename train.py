import argparse
import os
import torch
from tqdm import tqdm
import pdb

from pointpillars.utils import setup_seed
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars, MultimodalPointPillars
from pointpillars.loss import Loss
from torch.utils.tensorboard import SummaryWriter

def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)
    writer.flush()


def format_loss_log(loss_dict):
    loss_items = []
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            v = v.detach().cpu().item()
        loss_items.append(f'{k}: {v:.4f}')
    return ', '.join(loss_items)


def loss_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return float(value)


def move_data_to_cuda(data_dict):
    for key, value in data_dict.items():
        if torch.is_tensor(value):
            data_dict[key] = value.cuda()
        elif isinstance(value, list):
            for j, item in enumerate(value):
                if torch.is_tensor(item):
                    value[j] = item.cuda()


def atomic_torch_save(obj, path):
    tmp_path = path + '.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_training_state(path, model, optimizer, scheduler, completed_epoch,
                        args, extra_state=None):
    checkpoint = {
        'epoch': completed_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args),
    }
    if extra_state is not None:
        checkpoint.update(extra_state)
    atomic_torch_save(checkpoint, path)


def save_epoch_checkpoint(saved_ckpt_path, model, optimizer, scheduler,
                          completed_epoch, args, extra_state):
    atomic_torch_save(
        model.state_dict(),
        os.path.join(saved_ckpt_path, f'epoch_{completed_epoch}.pth')
    )
    save_training_state(
        os.path.join(saved_ckpt_path, f'epoch_{completed_epoch}_train_state.pth'),
        model, optimizer, scheduler, completed_epoch, args,
        extra_state=extra_state
    )


def load_model_weights(model, state_dict, strict=True):
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing = getattr(incompatible, 'missing_keys', [])
        unexpected = getattr(incompatible, 'unexpected_keys', [])
        if missing:
            print(f'Ignored missing checkpoint keys: {len(missing)}')
        if unexpected:
            print(f'Ignored unexpected checkpoint keys: {len(unexpected)}')


def load_training_checkpoint(args, model, optimizer, scheduler, saved_ckpt_path):
    resume_path = args.resume_from
    if args.auto_resume and resume_path is None:
        latest_path = os.path.join(saved_ckpt_path, 'latest_train_state.pth')
        if os.path.isfile(latest_path):
            resume_path = latest_path

    if resume_path is None:
        if args.start_epoch != 0:
            raise ValueError('--start_epoch requires --resume_from or --auto_resume')
        return 0, None

    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')

    device = torch.device('cpu' if args.no_cuda else 'cuda')
    checkpoint = torch.load(resume_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        strict = not (args.multimodal and args.resume_weights_only)
        load_model_weights(model, checkpoint['model_state_dict'], strict=strict)
        if not args.resume_weights_only:
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
        else:
            start_epoch = args.start_epoch
    else:
        strict = not (args.multimodal and args.resume_weights_only)
        load_model_weights(model, checkpoint, strict=strict)
        start_epoch = args.start_epoch
        if start_epoch == 0:
            print(
                'Loaded a weights-only checkpoint. Pass --start_epoch if this '
                'checkpoint was saved after an earlier training epoch.'
            )

    if args.start_epoch != 0:
        start_epoch = args.start_epoch

    print(f'Resumed training from {resume_path}; starting at epoch {start_epoch + 1}.')
    if isinstance(checkpoint, dict):
        return start_epoch, checkpoint
    return start_epoch, None


def add_sin_difference(bbox_pred, bbox_target):
    """Encode yaw residual without mutating either angle before both terms exist."""
    pred_yaw = bbox_pred[:, -1].clone()
    target_yaw = bbox_target[:, -1].clone()
    bbox_pred[:, -1] = torch.sin(pred_yaw) * torch.cos(target_yaw)
    bbox_target[:, -1] = torch.cos(pred_yaw) * torch.sin(target_yaw)
    return bbox_pred, bbox_target


def main(args):
    if args.early_stop_patience < 0:
        raise ValueError('--early_stop_patience must be >= 0')
    if args.early_stop_min_delta < 0:
        raise ValueError('--early_stop_min_delta must be >= 0')

    setup_seed()
    train_dataset = Kitti(data_root=args.data_root,
                          split='train',
                          multimodal=args.multimodal,
                          multimodal_aug=args.multimodal_aug)
    val_dataset = Kitti(data_root=args.data_root,
                        split='val',
                        multimodal=args.multimodal,
                        multimodal_aug=args.multimodal_aug)
    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)

    model_cls = MultimodalPointPillars if args.multimodal else PointPillars
    if not args.no_cuda:
        pointpillars = model_cls(nclasses=args.nclasses).cuda()
    else:
        pointpillars = model_cls(nclasses=args.nclasses)
    loss_func = Loss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    if args.multimodal:
        image_params = list(pointpillars.image_branch.parameters())
        fusion_params = list(pointpillars.pillar_encoder.image_proj.parameters()) + \
            list(pointpillars.pillar_encoder.image_gate.parameters())
        specialized_ids = {id(param) for param in image_params + fusion_params}
        lidar_params = [
            param for param in pointpillars.parameters()
            if id(param) not in specialized_ids
        ]
        optimizer_params = [
            {'params': image_params, 'lr': init_lr * 0.1, 'weight_decay': 1e-4},
            {'params': fusion_params, 'lr': init_lr, 'weight_decay': 1e-2},
            {'params': lidar_params, 'lr': init_lr * 0.4, 'weight_decay': 1e-2},
        ]
        max_lrs = [init_lr, init_lr * 10, init_lr * 4]
    else:
        optimizer_params = pointpillars.parameters()
        max_lrs = init_lr * 10
    optimizer = torch.optim.AdamW(params=optimizer_params,
                                  lr=init_lr,
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=max_lrs,
                                                    total_steps=max_iters, 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10)
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)

    start_epoch, resume_checkpoint = load_training_checkpoint(
        args, pointpillars, optimizer, scheduler, saved_ckpt_path
    )
    if start_epoch >= args.max_epoch:
        print(
            f'Start epoch {start_epoch} is already >= max_epoch '
            f'{args.max_epoch}; nothing to train.'
        )
        return

    best_val_loss = None
    best_epoch = 0
    early_stop_counter = 0
    if resume_checkpoint is not None:
        best_val_loss = resume_checkpoint.get('best_val_loss')
        best_epoch = resume_checkpoint.get('best_epoch', 0)
        early_stop_counter = resume_checkpoint.get('early_stop_counter', 0)

    for epoch in range(start_epoch, args.max_epoch):
        print('=' * 20, epoch, '=' * 20)
        train_step, val_step = 0, 0
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:
                move_data_to_cuda(data_dict)
            
            optimizer.zero_grad()

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batched_difficulty = data_dict['batched_difficulty']
            model_kwargs = dict(
                batched_pts=batched_pts,
                mode='train',
                batched_gt_bboxes=batched_gt_bboxes,
                batched_gt_labels=batched_labels,
            )
            if args.multimodal:
                model_kwargs.update(
                    batched_imgs=data_dict['batched_imgs'],
                    batched_img_info=data_dict['batched_img_info'],
                    batched_calib_info=data_dict['batched_calib_info'],
                )
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = pointpillars(**model_kwargs)
            
            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
            batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
            batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
            # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
            batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
            # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
            
            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
            class_pos_counts = [
                (batched_bbox_labels == class_id).sum()
                for class_id in range(args.nclasses)
            ]
            bbox_pred = bbox_pred[pos_idx]
            batched_bbox_reg = batched_bbox_reg[pos_idx]
            bbox_pred, batched_bbox_reg = add_sin_difference(
                bbox_pred, batched_bbox_reg
            )
            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            batched_dir_labels = batched_dir_labels[pos_idx]

            num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

            loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                  bbox_pred=bbox_pred,
                                  bbox_dir_cls_pred=bbox_dir_cls_pred,
                                  batched_labels=batched_bbox_labels, 
                                  num_cls_pos=num_cls_pos, 
                                  batched_bbox_reg=batched_bbox_reg, 
                                  batched_dir_labels=batched_dir_labels)
            
            loss = loss_dict['total_loss']
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
            optimizer.step()
            scheduler.step()

            global_step = epoch * len(train_dataloader) + train_step + 1

            if global_step % args.log_freq == 0:
                save_summary(writer, loss_dict, global_step, 'train',
                             lr=optimizer.param_groups[0]['lr'], 
                             momentum=optimizer.param_groups[0]['betas'][0])
                if args.multimodal:
                    for name, value in pointpillars.pillar_encoder.last_fusion_stats.items():
                        writer.add_scalar(f'fusion/{name}', value, global_step)
                for class_name, class_id in Kitti.CLASSES.items():
                    writer.add_scalar(
                        f'anchors/positive_{class_name.lower()}',
                        class_pos_counts[class_id],
                        global_step,
                    )
                tqdm.write(
                    f'[train] epoch: {epoch + 1}/{args.max_epoch}, '
                    f'step: {global_step}, {format_loss_log(loss_dict)}'
                )
            train_step += 1

        completed_epoch = epoch + 1
        latest_state_path = os.path.join(saved_ckpt_path, 'latest_train_state.pth')
        extra_state = {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'early_stop_counter': early_stop_counter,
        }
        save_training_state(latest_state_path, pointpillars, optimizer,
                            scheduler, completed_epoch, args,
                            extra_state=extra_state)
        if completed_epoch % args.ckpt_freq_epoch == 0 or completed_epoch == args.max_epoch:
            save_epoch_checkpoint(saved_ckpt_path, pointpillars, optimizer,
                                  scheduler, completed_epoch, args,
                                  extra_state)

        if epoch % 2 == 0:
            continue
        pointpillars.eval()
        val_loss_sum = 0.0
        val_loss_count = 0
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                if not args.no_cuda:
                    move_data_to_cuda(data_dict)
                
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                batched_difficulty = data_dict['batched_difficulty']
                model_kwargs = dict(
                    batched_pts=batched_pts,
                    mode='train',
                    batched_gt_bboxes=batched_gt_bboxes,
                    batched_gt_labels=batched_labels,
                )
                if args.multimodal:
                    model_kwargs.update(
                        batched_imgs=data_dict['batched_imgs'],
                        batched_img_info=data_dict['batched_img_info'],
                        batched_calib_info=data_dict['batched_calib_info'],
                    )
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = pointpillars(**model_kwargs)
                
                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
                
                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                bbox_pred, batched_bbox_reg = add_sin_difference(
                    bbox_pred, batched_bbox_reg
                )
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                    bbox_pred=bbox_pred,
                                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                                    batched_labels=batched_bbox_labels, 
                                    num_cls_pos=num_cls_pos, 
                                    batched_bbox_reg=batched_bbox_reg, 
                                    batched_dir_labels=batched_dir_labels)
                val_loss_sum += loss_value(loss_dict['total_loss'])
                val_loss_count += 1
                
                global_step = epoch * len(val_dataloader) + val_step + 1
                if global_step % args.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, 'val')
                    tqdm.write(
                        f'[val] epoch: {epoch + 1}/{args.max_epoch}, '
                        f'step: {global_step}, {format_loss_log(loss_dict)}'
                    )
                val_step += 1
        pointpillars.train()

        if val_loss_count == 0:
            continue

        mean_val_loss = val_loss_sum / val_loss_count
        writer.add_scalar('val/epoch_total_loss', mean_val_loss, completed_epoch)
        writer.flush()
        if args.early_stop_patience <= 0:
            continue

        improved = (
            best_val_loss is None or
            mean_val_loss < best_val_loss - args.early_stop_min_delta
        )
        if improved:
            best_val_loss = mean_val_loss
            best_epoch = completed_epoch
            early_stop_counter = 0
            extra_state = {
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'early_stop_counter': early_stop_counter,
            }
            atomic_torch_save(
                pointpillars.state_dict(),
                os.path.join(saved_ckpt_path, 'best.pth')
            )
            save_training_state(
                os.path.join(saved_ckpt_path, 'best_train_state.pth'),
                pointpillars, optimizer, scheduler, completed_epoch, args,
                extra_state=extra_state
            )
            tqdm.write(
                f'[early-stop] best val total_loss improved to '
                f'{best_val_loss:.4f} at epoch {best_epoch}'
            )
        else:
            early_stop_counter += 1
            tqdm.write(
                f'[early-stop] val total_loss {mean_val_loss:.4f} did not '
                f'improve from {best_val_loss:.4f}; '
                f'{early_stop_counter}/{args.early_stop_patience}'
            )

        extra_state = {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'early_stop_counter': early_stop_counter,
        }
        save_training_state(latest_state_path, pointpillars, optimizer,
                            scheduler, completed_epoch, args,
                            extra_state=extra_state)
        if early_stop_counter >= args.early_stop_patience:
            save_epoch_checkpoint(saved_ckpt_path, pointpillars, optimizer,
                                  scheduler, completed_epoch, args,
                                  extra_state)
            print(
                f'Early stopping at epoch {completed_epoch}. '
                f'Best val total_loss {best_val_loss:.4f} was at epoch {best_epoch}.'
            )
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--focal_alpha', type=float, nargs=3,
                        default=[0.50, 0.35, 0.25],
                        metavar=('PEDESTRIAN', 'CYCLIST', 'CAR'),
                        help='positive focal-loss alpha for each class')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='focal-loss focusing exponent')
    parser.add_argument('--max_epoch', type=int, default=160)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=10)
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='validation checks without val total_loss improvement before stopping; 0 disables')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='minimum val total_loss improvement required to reset early stopping')
    parser.add_argument('--resume_from', default=None,
                        help='path to a checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='completed epoch for weights-only checkpoints')
    parser.add_argument('--auto_resume', action='store_true',
                        help='resume from saved_path/checkpoints/latest_train_state.pth if it exists')
    parser.add_argument('--resume_weights_only', action='store_true',
                        help='load only model weights, ignoring optimizer and scheduler state')
    parser.add_argument('--multimodal', action='store_true',
                        help='enable image-lidar early fusion model')
    parser.add_argument('--multimodal_aug',
                        choices=['image_aligned', 'full_lidar', 'none'],
                        default='image_aligned',
                        help='training augmentation policy')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
