import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.reduction    = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        p_t        = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha      = self.alpha.to(targets.device)
            alpha_t    = alpha[targets]
            alpha_t[targets == self.ignore_index] = 1.0
            focal_loss = alpha_t * focal_loss
        
        focal_loss[targets == self.ignore_index] = 0.0
        
        if self.reduction == 'mean':
            non_ignore = (targets != self.ignore_index).sum()
            return focal_loss.sum() / max(non_ignore, 1)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss