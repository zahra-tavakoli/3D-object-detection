import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, alpha=(0.50, 0.35, 0.25), gamma=2.0, beta=1/9,
                 cls_w=1.0, reg_w=2.0, dir_w=0.2):
        super().__init__()
        alpha = torch.as_tensor(alpha, dtype=torch.float32)
        if alpha.ndim > 1 or torch.any((alpha <= 0) | (alpha >= 1)):
            raise ValueError('alpha must contain values strictly between 0 and 1')
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.cls_w = cls_w
        self.reg_w = reg_w
        self.dir_w = dir_w
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none',
                                              beta=beta)
        self.dir_cls = nn.CrossEntropyLoss()
    
    def forward(self,
                bbox_cls_pred,
                bbox_pred,
                bbox_dir_cls_pred,
                batched_labels, 
                num_cls_pos, 
                batched_bbox_reg, 
                batched_dir_labels):
        '''
        bbox_cls_pred: (n, 3)
        bbox_pred: (n, 7)
        bbox_dir_cls_pred: (n, 2)
        batched_labels: (n, )
        num_cls_pos: int
        batched_bbox_reg: (n, 7)
        batched_dir_labels: (n, )
        return: loss, float.
        '''
        # 1. bbox cls loss
        # focal loss: FL = - \alpha_t (1 - p_t)^\gamma * log(p_t)
        #             y == 1 -> p_t = p
        #             y == 0 -> p_t = 1 - p
        nclasses = bbox_cls_pred.size(1)
        batched_labels = F.one_hot(batched_labels, nclasses + 1)[:, :nclasses].float() # (n, 3)

        if self.alpha.numel() not in (1, nclasses):
            raise ValueError(
                f'Expected one focal alpha or {nclasses} class alphas, '
                f'got {self.alpha.numel()}'
            )
        alpha = self.alpha.to(
            device=bbox_cls_pred.device,
            dtype=bbox_cls_pred.dtype,
        ).reshape(1, -1)

        bbox_cls_pred_sigmoid = torch.sigmoid(bbox_cls_pred)
        weights = alpha * (1 - bbox_cls_pred_sigmoid).pow(self.gamma) * batched_labels + \
             (1 - alpha) * bbox_cls_pred_sigmoid.pow(self.gamma) * (1 - batched_labels) # (n, 3)
        cls_loss = F.binary_cross_entropy(bbox_cls_pred_sigmoid, batched_labels, reduction='none')
        cls_loss = cls_loss * weights
        num_cls_pos = torch.clamp(num_cls_pos.float(), min=1.0)
        cls_loss = cls_loss.sum() / num_cls_pos

        # 2. regression loss
        if bbox_pred.numel() == 0:
            reg_loss = bbox_cls_pred.sum() * 0
        else:
            reg_loss = self.smooth_l1_loss(bbox_pred, batched_bbox_reg)
            reg_loss = reg_loss.sum() / reg_loss.size(0)

        # 3. direction cls loss
        if bbox_dir_cls_pred.numel() == 0:
            dir_cls_loss = bbox_cls_pred.sum() * 0
        else:
            dir_cls_loss = self.dir_cls(bbox_dir_cls_pred, batched_dir_labels)

        # 4. total loss
        total_loss = self.cls_w * cls_loss + self.reg_w * reg_loss + self.dir_w * dir_cls_loss
        
        loss_dict={'cls_loss': cls_loss, 
                   'reg_loss': reg_loss,
                   'dir_cls_loss': dir_cls_loss,
                   'total_loss': total_loss}
        return loss_dict
