import argparse
import os
import torch
from tqdm import tqdm

from pointpillars.utils import setup_seed
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.dataset import V2X
from pointpillars.model import PointPillars, MultimodalPointPillars
from pointpillars.loss import Loss
from torch.utils.tensorboard import SummaryWriter


def save_summary(writer, loss_dict, global_step, tag, lr=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)


def move_to_cuda(data_dict):
    """Move data to GPU"""
    for key in data_dict:
        if key in ['batched_pts', 'batched_gt_bboxes', 'batched_labels', 'batched_difficulty']:
            for j in range(len(data_dict[key])):
                if torch.is_tensor(data_dict[key][j]):
                    data_dict[key][j] = data_dict[key][j].cuda()
        elif key == 'batched_images' and data_dict[key] is not None:
            data_dict[key] = data_dict[key].cuda()
    return data_dict


def train_one_epoch(model, dataloader, optimizer, scheduler, loss_func, 
                    nclasses, epoch, writer, args, use_multimodal=False):
    model.train()
    train_step = 0
    
    tbar = tqdm(dataloader, total=len(dataloader), ncols=120, leave=False)
    for i, data_dict in enumerate(tbar):
        if not args.no_cuda:
            data_dict = move_to_cuda(data_dict)
        
        optimizer.zero_grad()
        
        batched_pts = data_dict['batched_pts']
        batched_gt_bboxes = data_dict['batched_gt_bboxes']
        batched_labels = data_dict['batched_labels']
        
        # Forward
        if use_multimodal:
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = model(
                batched_pts=batched_pts,
                mode='train',
                batched_gt_bboxes=batched_gt_bboxes,
                batched_gt_labels=batched_labels,
                batched_images=data_dict.get('batched_images'),
                batched_calib=data_dict.get('batched_calib_info')
            )
        else:
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = model(
                batched_pts=batched_pts,
                mode='train',
                batched_gt_bboxes=batched_gt_bboxes,
                batched_gt_labels=batched_labels
            )
        
        # Prepare predictions
        bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, nclasses)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

        batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
        batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
        batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
        batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)

        pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < nclasses)
        bbox_pred = bbox_pred[pos_idx]
        batched_bbox_reg = batched_bbox_reg[pos_idx]

        # Angle encoding
        bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
        batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())

        bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
        batched_dir_labels = batched_dir_labels[pos_idx]

        num_cls_pos = (batched_bbox_labels < nclasses).sum()
        bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
        batched_bbox_labels[batched_bbox_labels < 0] = nclasses
        batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

        # Loss
        loss_dict = loss_func(
            bbox_cls_pred=bbox_cls_pred,
            bbox_pred=bbox_pred,
            bbox_dir_cls_pred=bbox_dir_cls_pred,
            batched_labels=batched_bbox_labels,
            num_cls_pos=num_cls_pos,
            batched_bbox_reg=batched_bbox_reg,
            batched_dir_labels=batched_dir_labels
        )

        loss = loss_dict['total_loss']
        
        tbar.set_postfix({
            'loss': f"{float(loss):.3f}",
            'cls': f"{float(loss_dict['cls_loss']):.3f}",
            'reg': f"{float(loss_dict['reg_loss']):.3f}",
            'dir': f"{float(loss_dict['dir_cls_loss']):.3f}",
        })

        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step = epoch * len(dataloader) + train_step + 1
        if global_step % args.log_freq == 0:
            save_summary(writer, loss_dict, global_step, 'train', 
                        lr=optimizer.param_groups[0]['lr'])
        train_step += 1


def main(args):
    setup_seed()
    
    # Dataset
    if args.dataset == 'V2X':
        train_dataset = V2X(data_root=args.data_root, split='train', load_image=args.multimodal)
        val_dataset = V2X(data_root=args.data_root, split='val', load_image=args.multimodal)
        CLASSES = V2X.CLASSES
        dataset_name = 'v2x'
    else:
        train_dataset = Kitti(data_root=args.data_root, split='train')
        val_dataset = Kitti(data_root=args.data_root, split='val')
        CLASSES = Kitti.CLASSES
        dataset_name = 'kitti'
    
    nclasses = len(CLASSES) if args.nclasses is None else args.nclasses
    
    train_dataloader = get_dataloader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True, drop_last=True
    )
    val_dataloader = get_dataloader(
        val_dataset, args.batch_size, args.num_workers, shuffle=False
    )
    
    # Model
    if args.multimodal:
        print(f"Using Multimodal PointPillars with {args.image_backbone}")
        model = MultimodalPointPillars(
            nclasses=nclasses,
            dataset=dataset_name,
            image_backbone=args.image_backbone,
            image_dim=64,
            pretrained=True
        )
    else:
        print("Using LiDAR-only PointPillars")
        model = PointPillars(nclasses=nclasses, dataset=dataset_name)
    
    if not args.no_cuda:
        model = model.cuda()
    
    loss_func = Loss()
    
    # Optimizer
    max_iters = len(train_dataloader) * args.max_epoch
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.init_lr, betas=(0.95, 0.99), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.init_lr*10, total_steps=max_iters,
        pct_start=0.4, anneal_strategy='cos',
        cycle_momentum=True, base_momentum=0.95*0.895, max_momentum=0.95, div_factor=10
    )
    
    # Logging
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)
    
    print(f"Training on {dataset_name.upper()} with {nclasses} classes")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training loop
    for epoch in range(args.max_epoch):
        print('=' * 20, f'Epoch {epoch}', '=' * 20)
        
        train_one_epoch(
            model, train_dataloader, optimizer, scheduler, loss_func,
            nclasses, epoch, writer, args, use_multimodal=args.multimodal
        )
        
        # Save checkpoint
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            ckpt_path = os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
    
    # Save final
    torch.save(model.state_dict(), os.path.join(saved_ckpt_path, 'final.pth'))
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointPillars Training')
    parser.add_argument('--data_root', required=True, help='Dataset root path')
    parser.add_argument('--dataset', default='V2X', choices=['kitti', 'V2X'])
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--nclasses', type=int, default=None)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)
    parser.add_argument('--no_cuda', action='store_true')
    
    # Multimodal options
    parser.add_argument('--multimodal', action='store_true', 
                        help='Use multimodal (LiDAR + Camera) fusion')
    parser.add_argument('--image_backbone', default='resnet18', 
                        choices=['resnet18', 'resnet50'])
    
    args = parser.parse_args()
    main(args)