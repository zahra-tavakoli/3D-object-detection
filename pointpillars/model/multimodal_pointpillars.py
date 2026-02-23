# pointpillars/model/multimodal_pointpillars.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# torchvision 0.9.1 compatible imports
from torchvision.models import resnet18, resnet50

from .pointpillars import (
    PointPillars, PillarEncoder, PillarLayer, Backbone, Neck, Head,
    get_model_config
)
from .anchors import Anchors, anchor_target


class ImageBackbone(nn.Module):
    """ResNet backbone for image feature extraction (torchvision 0.9.1 compatible)"""
    
    def __init__(self, name="resnet18", pretrained=True):
        super().__init__()
        
        # === FIX: Use old API for torchvision 0.9.1 ===
        if name == "resnet18":
            backbone = resnet18(pretrained=pretrained)
            self.out_channels = 256  # layer3 output
        elif name == "resnet50":
            backbone = resnet50(pretrained=pretrained)
            self.out_channels = 1024  # layer3 output
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        
        # Extract layers up to layer3
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        # Skip layer4 for efficiency
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x


class ImageEncoder(nn.Module):
    """Image encoder: ResNet backbone + channel reduction"""
    
    def __init__(self, backbone_name="resnet18", out_channels=64, pretrained=True):
        super().__init__()
        
        self.backbone = ImageBackbone(backbone_name, pretrained)
        
        self.reducer = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out_channels = out_channels
    
    def forward(self, x):
        """
        Args:
            x: (bs, 3, H, W) RGB image [0-1 normalized]
        Returns:
            (bs, out_channels, H', W') feature map
        """
        feat = self.backbone(x)
        feat = self.reducer(feat)
        return feat


class PillarFusion(nn.Module):
    """Fusion module for LiDAR-Camera feature fusion"""
    
    def __init__(self, point_dim, image_dim):
        super().__init__()
        
        if point_dim != image_dim:
            self.align = nn.Linear(image_dim, point_dim)
        else:
            self.align = None
    
    def project_to_image(self, points, P2, R0_rect, Tr_velo_to_cam, image_shape):
        """Project 3D LiDAR points to 2D image coordinates"""
        device = points.device
        dtype = points.dtype
        N = points.shape[0]
        
        ones = torch.ones(N, 1, device=device, dtype=dtype)
        pts_hom = torch.cat([points, ones], dim=1)
        
        pts_cam = pts_hom @ Tr_velo_to_cam.T
        pts_rect = pts_cam @ R0_rect.T
        pts_img = pts_rect @ P2.T
        
        depth = pts_img[:, 2:3].clamp(min=1e-5)
        uv = pts_img[:, :2] / depth
        
        H, W = image_shape
        valid = (
            (pts_img[:, 2] > 0) &
            (uv[:, 0] >= 0) & (uv[:, 0] < W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < H)
        )
        
        return uv, valid
    
    def sample_features(self, img_feat, uv, valid, batch_idx, bs):
        """Sample image features at projected locations"""
        _, C, H, W = img_feat.shape
        device = img_feat.device
        dtype = img_feat.dtype
        N = uv.shape[0]
        
        sampled = torch.zeros(N, C, device=device, dtype=dtype)
        
        for b in range(bs):
            mask = (batch_idx == b) & valid
            if mask.sum() == 0:
                continue
            
            uv_b = uv[mask]
            
            grid_x = 2.0 * uv_b[:, 0] / (W - 1) - 1.0
            grid_y = 2.0 * uv_b[:, 1] / (H - 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.view(1, 1, -1, 2)
            
            feat_b = img_feat[b:b+1]
            sampled_b = F.grid_sample(
                feat_b, grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=True
            )
            sampled_b = sampled_b.view(C, -1).T
            
            sampled[mask] = sampled_b
        
        return sampled
    
    def forward(self, point_feat, pillar_centers, batch_idx, 
                img_feat, batched_calib, img_shape):
        """Fuse point features with image features"""
        device = point_feat.device
        dtype = point_feat.dtype
        bs = img_feat.shape[0]
        N = point_feat.shape[0]
        
        all_uv = torch.zeros(N, 2, device=device, dtype=dtype)
        all_valid = torch.zeros(N, dtype=torch.bool, device=device)
        
        for b in range(bs):
            mask = batch_idx == b
            if mask.sum() == 0:
                continue
            
            pts_b = pillar_centers[mask]
            calib = batched_calib[b]
            
            if isinstance(calib['P2'], np.ndarray):
                P2 = torch.from_numpy(calib['P2']).to(device=device, dtype=dtype)
                R0 = torch.from_numpy(calib['R0_rect']).to(device=device, dtype=dtype)
                Tr = torch.from_numpy(calib['Tr_velo_to_cam']).to(device=device, dtype=dtype)
            else:
                P2 = calib['P2'].to(device=device, dtype=dtype)
                R0 = calib['R0_rect'].to(device=device, dtype=dtype)
                Tr = calib['Tr_velo_to_cam'].to(device=device, dtype=dtype)
            
            uv, valid = self.project_to_image(pts_b, P2, R0, Tr, img_shape)
            all_uv[mask] = uv
            all_valid[mask] = valid
        
        img_sampled = self.sample_features(img_feat, all_uv, all_valid, batch_idx, bs)
        
        if self.align is not None:
            img_sampled = self.align(img_sampled)
        
        fused = point_feat + img_sampled
        
        return fused


class MultimodalPillarEncoder(PillarEncoder):
    """Pillar encoder with image fusion"""
    
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel, image_dim):
        super().__init__(voxel_size, point_cloud_range, in_channel, out_channel)
        self.fusion = PillarFusion(out_channel, image_dim)
    
    def forward(self, pillars, coors_batch, npoints_per_pillar,
                img_feat=None, batched_calib=None, img_shape=None):
        device = pillars.device
        
        # Standard pillar encoding
        offset_pt_center = pillars[:, :, :3] - torch.sum(
            pillars[:, :, :3], dim=1, keepdim=True
        ) / npoints_per_pillar[:, None, None]

        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset)

        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1)
        features[:, :, 0:1] = x_offset_pi_center
        features[:, :, 1:2] = y_offset_pi_center

        voxel_ids = torch.arange(0, pillars.size(1)).to(device)
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]
        mask = mask.permute(1, 0).contiguous()
        features *= mask[:, :, None]

        features = features.permute(0, 2, 1).contiguous()
        features = F.relu(self.bn(self.conv(features)))
        pooling_features = torch.max(features, dim=-1)[0]
        
        # Image fusion
        if img_feat is not None and batched_calib is not None and img_shape is not None:
            pillar_centers = torch.stack([
                coors_batch[:, 1].float() * self.vx + self.x_offset,
                coors_batch[:, 2].float() * self.vy + self.y_offset,
                torch.zeros(coors_batch.shape[0], device=device)
            ], dim=1)
            
            batch_idx = coors_batch[:, 0]
            
            pooling_features = self.fusion(
                pooling_features, pillar_centers, batch_idx,
                img_feat, batched_calib, img_shape
            )
        
        # Scatter to BEV
        batched_canvas = []
        bs = int(coors_batch[-1, 0].item()) + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), 
                                dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1].long(), cur_coors[:, 2].long()] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        
        return torch.stack(batched_canvas, dim=0)


class MultimodalPointPillars(PointPillars):
    """Multimodal PointPillars with LiDAR-Camera fusion"""
    
    def __init__(self, nclasses, dataset='kitti', config=None,
                 image_backbone="resnet18", image_dim=64, pretrained=True):
        # Initialize nn.Module
        nn.Module.__init__(self)
        
        if config is None:
            config = get_model_config(dataset)
        self.config = config
        self.nclasses = nclasses
        
        # Pillar layer
        self.pillar_layer = PillarLayer(
            voxel_size=config['voxel_size'],
            point_cloud_range=config['point_cloud_range'],
            max_num_points=config['max_num_points'],
            max_voxels=config['max_voxels']
        )
        
        # Multimodal pillar encoder
        self.pillar_encoder = MultimodalPillarEncoder(
            voxel_size=config['voxel_size'],
            point_cloud_range=config['point_cloud_range'],
            in_channel=9,
            out_channel=64,
            image_dim=image_dim
        )
        
        # Image encoder
        self.image_encoder = ImageEncoder(
            backbone_name=image_backbone,
            out_channels=image_dim,
            pretrained=pretrained
        )
        
        # 2D backbone
        self.backbone = Backbone(
            in_channel=64, 
            out_channels=[64, 128, 256], 
            layer_nums=[3, 5, 5]
        )
        
        # Neck
        self.neck = Neck(
            in_channels=[64, 128, 256], 
            upsample_strides=[1, 2, 4], 
            out_channels=[128, 128, 128]
        )
        
        # Anchors
        self.anchors_generator = Anchors(
            ranges=config['anchor_ranges'],
            sizes=config['anchor_sizes'],
            rotations=config['anchor_rotations']
        )
        
        # Head
        n_anchors = len(config['anchor_sizes']) * len(config['anchor_rotations'])
        self.head = Head(in_channel=384, n_anchors=n_anchors, n_classes=nclasses)
        self.assigners = config['assigners']

        # Inference params
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50
    
    def forward(self, batched_pts, mode='test', 
                batched_gt_bboxes=None, batched_gt_labels=None,
                batched_images=None, batched_calib=None):
        batch_size = len(batched_pts)
        
        # Image features
        img_feat = None
        img_shape = None
        if batched_images is not None:
            img_feat = self.image_encoder(batched_images)
            img_shape = (batched_images.shape[2], batched_images.shape[3])
        
        # Pillarization
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        
        # Pillar encoding with fusion
        pillar_features = self.pillar_encoder(
            pillars, coors_batch, npoints_per_pillar,
            img_feat, batched_calib, img_shape
        )
        
        # Backbone + Neck + Head
        xs = self.backbone(pillar_features)
        x = self.neck(xs)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)
        
        # Anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]
        
        if mode == 'train':
            anchor_target_dict = anchor_target(
                batched_anchors=batched_anchors,
                batched_gt_bboxes=batched_gt_bboxes,
                batched_gt_labels=batched_gt_labels,
                assigners=self.assigners,
                nclasses=self.nclasses
            )
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        
        elif mode in ['val', 'test']:
            results = self.get_predicted_bboxes(
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors
            )
            return results
        else:
            raise ValueError(f"Invalid mode: {mode}")