import torch
import torch.nn as nn
import torch.nn.functional as F

from pointpillars.model.anchors import anchor_target
from pointpillars.model.pointpillars import PointPillars, PillarEncoder


class ImageFeatureNet(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        try:
            from torchvision.models import resnet18
        except ImportError as exc:
            raise ImportError(
                'MultimodalPointPillars requires torchvision. Use the 3detection '
                'conda environment or install a torchvision version compatible '
                'with the current torch build.'
            ) from exc

        resnet = resnet18(pretrained=True)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.lateral1 = nn.Conv2d(64, out_channels, 1)
        self.lateral2 = nn.Conv2d(128, out_channels, 1)
        self.lateral3 = nn.Conv2d(256, out_channels, 1)
        self.lateral4 = nn.Conv2d(512, out_channels, 1)
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, imgs):
        x = self.stem(imgs)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lateral4(c5)
        p4 = self.lateral3(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral2(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral1(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        return self.smooth(p2)


class MultimodalPillarEncoder(PillarEncoder):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel,
                 image_drop_prob=0.2, init_image_gate=-4.0):
        super().__init__(voxel_size, point_cloud_range, in_channel, out_channel)
        self.image_proj = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, 1, bias=False),
            nn.ReLU(inplace=False),
        )
        self.image_dropout = nn.Dropout(p=image_drop_prob)
        self.image_gate = nn.Parameter(torch.full((1, out_channel, 1), init_image_gate))
        self.init_image_projection()

    def init_image_projection(self):
        proj = self.image_proj[0]
        nn.init.normal_(proj.weight, mean=0.0, std=0.001)

    def sample_image_features(self, image_features, pillars, npoints_per_pillar,
                              coors_batch, batched_calib_info, batched_img_info):
        device = pillars.device
        dtype = pillars.dtype
        n_pillars, max_points = pillars.shape[:2]
        sampled_features = pillars.new_zeros((n_pillars, max_points, self.out_channel))
        point_ids = torch.arange(max_points, device=device)

        for batch_id in range(image_features.size(0)):
            pillar_mask = coors_batch[:, 0] == batch_id
            if pillar_mask.sum() == 0:
                continue

            cur_pillars = pillars[pillar_mask]
            cur_npoints = npoints_per_pillar[pillar_mask]
            valid_points = point_ids[None, :] < cur_npoints[:, None]
            if valid_points.sum() == 0:
                continue

            if cur_pillars.size(-1) >= 8:
                flat_points = cur_pillars[:, :, 4:7].reshape(-1, 3)
                image_valid = cur_pillars[:, :, 7] > 0.5
                flat_valid = (valid_points & image_valid).reshape(-1)
            else:
                flat_points = cur_pillars[:, :, :3].reshape(-1, 3)
                flat_valid = valid_points.reshape(-1)

            calib_info = batched_calib_info[batch_id]
            p2 = torch.as_tensor(calib_info['P2'], dtype=dtype, device=device)
            r0_rect = torch.as_tensor(calib_info['R0_rect'], dtype=dtype, device=device)
            tr_velo_to_cam = torch.as_tensor(calib_info['Tr_velo_to_cam'], dtype=dtype, device=device)
            lidar_to_img = p2 @ r0_rect @ tr_velo_to_cam

            ones = torch.ones((flat_points.size(0), 1), dtype=dtype, device=device)
            flat_points_homo = torch.cat([flat_points, ones], dim=1)
            projected = flat_points_homo @ lidar_to_img.t()
            depth = projected[:, 2]
            eps = torch.tensor(1e-5, dtype=dtype, device=device)
            uv = projected[:, :2] / torch.clamp(depth[:, None], min=eps)

            image_shape = batched_img_info[batch_id]['image_shape']
            resized_shape = batched_img_info[batch_id]['resized_shape']
            orig_h, orig_w = float(image_shape[0]), float(image_shape[1])
            resized_h, resized_w = float(resized_shape[0]), float(resized_shape[1])

            x = uv[:, 0] * (resized_w - 1.0) / max(orig_w - 1.0, 1.0)
            y = uv[:, 1] * (resized_h - 1.0) / max(orig_h - 1.0, 1.0)
            in_image = (
                (depth > eps) &
                (uv[:, 0] >= 0.0) & (uv[:, 0] <= orig_w - 1.0) &
                (uv[:, 1] >= 0.0) & (uv[:, 1] <= orig_h - 1.0) &
                flat_valid
            )
            if in_image.sum() == 0:
                continue

            grid_x = x[in_image] / max(resized_w - 1.0, 1.0) * 2.0 - 1.0
            grid_y = y[in_image] / max(resized_h - 1.0, 1.0) * 2.0 - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
            cur_img_features = F.grid_sample(
                image_features[batch_id:batch_id + 1],
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            )
            cur_img_features = cur_img_features.squeeze(0).squeeze(-1).t()

            cur_sampled = cur_pillars.new_zeros((cur_pillars.size(0) * max_points, self.out_channel))
            cur_sampled[in_image] = cur_img_features
            sampled_features[pillar_mask] = cur_sampled.view(cur_pillars.size(0), max_points, self.out_channel)

        return sampled_features

    def forward(self, pillars, coors_batch, npoints_per_pillar, image_features,
                batched_calib_info, batched_img_info):
        device = pillars.device
        lidar_pillars = pillars[:, :, :4]
        xyz = lidar_pillars[:, :, :3]
        offset_pt_center = xyz - torch.sum(
            xyz, dim=1, keepdim=True
        ) / npoints_per_pillar[:, None, None]

        x_offset_pi_center = xyz[:, :, :1] - (
            coors_batch[:, None, 1:2] * self.vx + self.x_offset
        )
        y_offset_pi_center = xyz[:, :, 1:2] - (
            coors_batch[:, None, 2:3] * self.vy + self.y_offset
        )

        features = torch.cat(
            [lidar_pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center],
            dim=-1,
        )
        features[:, :, 0:1] = x_offset_pi_center
        features[:, :, 1:2] = y_offset_pi_center

        voxel_ids = torch.arange(0, pillars.size(1), device=device)
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]
        mask = mask.permute(1, 0).contiguous()
        features *= mask[:, :, None]

        point_features = features.permute(0, 2, 1).contiguous()
        point_features = F.relu(self.bn(self.conv(point_features)))
        image_point_features = self.sample_image_features(
            image_features=image_features,
            pillars=pillars,
            npoints_per_pillar=npoints_per_pillar,
            coors_batch=coors_batch,
            batched_calib_info=batched_calib_info,
            batched_img_info=batched_img_info,
        ).permute(0, 2, 1).contiguous()
        image_point_features = self.image_dropout(image_point_features)
        image_update = self.image_proj(image_point_features)
        point_features = point_features + torch.sigmoid(self.image_gate) * image_update
        point_features = point_features * mask[:, None, :]
        pooling_features = torch.max(point_features, dim=-1)[0]

        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        return torch.stack(batched_canvas, dim=0)


class MultimodalPointPillars(PointPillars):
    def __init__(self,
                 nclasses=3,
                 voxel_size=[0.16, 0.16, 4],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points=32,
                 max_voxels=(16000, 40000)):
        super().__init__(
            nclasses=nclasses,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )
        self.image_branch = ImageFeatureNet(out_channels=64)
        self.pillar_encoder = MultimodalPillarEncoder(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            in_channel=9,
            out_channel=64,
        )

    def forward(self, batched_pts, batched_imgs=None, batched_img_info=None,
                batched_calib_info=None, mode='test', batched_gt_bboxes=None,
                batched_gt_labels=None):
        if batched_imgs is None or batched_img_info is None or batched_calib_info is None:
            raise ValueError('MultimodalPointPillars requires images, image info, and calibration info.')

        batch_size = len(batched_pts)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        image_features = self.image_branch(batched_imgs)
        pillar_features = self.pillar_encoder(
            pillars=pillars,
            coors_batch=coors_batch,
            npoints_per_pillar=npoints_per_pillar,
            image_features=image_features,
            batched_calib_info=batched_calib_info,
            batched_img_info=batched_img_info,
        )

        xs = self.backbone(pillar_features)
        x = self.neck(xs)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

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
                nclasses=self.nclasses,
            )
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        elif mode in ['val', 'test']:
            return self.get_predicted_bboxes(
                bbox_cls_pred=bbox_cls_pred,
                bbox_pred=bbox_pred,
                bbox_dir_cls_pred=bbox_dir_cls_pred,
                batched_anchors=batched_anchors,
            )
        else:
            raise ValueError
