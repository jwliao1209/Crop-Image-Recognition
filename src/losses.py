import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, output_avg=True, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.output_avg = output_avg

    def forward(self, x, y, sample_weight=None):
        if x.dim() > 2:
            x = x.view(x.size(0), x.size(1), -1)     # (B, C, H, W) -> (B, C, H*W)
            x = x.transpose(1, 2)                    # (B, C, H*W)  -> (B, H*W, C)
            x = x.contiguous().view(-1, x.size(2))   # (B, H*W, C)  -> (B*H*W, C)

        y = y.view(-1, 1)
        log_pt = F.log_softmax(x, dim=1)
        log_pt = log_pt.gather(1, y)
        log_pt = log_pt.view(-1)
        pt = log_pt.exp()
        loss = -(1-pt) ** self.gamma * log_pt

        if sample_weight is not None:
            loss *= sample_weight[y.long()].squeeze()

        return loss.mean() if self.output_avg else loss.sum()
