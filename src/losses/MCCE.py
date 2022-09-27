import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MCCE_Loss(nn.Module):
    '''
    implement CrossEntropy + Mutual channel loss
    The loss has 3 main component:
        L_CE(preds, targets) : cross entropy loss
        L_dis(feats, targets): Discriminality Component
        L_div(feats)         : Diversity Component
    The total loss is given by
        Loss = L_CE + mu * (L_dis - lambda * L_div)
    params:
        cnums: number of channels that represent 1 class
        cgroups[i]: number of classes that use cnums[i] channel to represent
        p: masking probability
        lambda, mu: the coefficient (weight) of the loss
    '''
    def __init__(self, num_class=219, cnums=(8, 9), cgroups=(179, 40),
                 p=0.4, lambda_=10, mu_=0.01):
        super().__init__()
        self.num_class = num_class
        self.cnums = cnums
        self.cgroups = cgroups
        self.p = p
        self.lambda_ = lambda_
        self.mu_ = mu_

        # construct splitting point (convient to do indexing)
        tmp = [n*g for n, g in zip(cnums, cgroups)]
        self.sp = [0] + [sum(tmp[:i+1]) for i in range(len(tmp))]
        # tmp looks like (1520, 484) and self.sp looks like [0,1520,2048]

        # construct some functions
        self.celoss = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, preds, feats, targets):
        '''
        B: batch size, C: channels, H: height, W: width
        params:
            feats: features,    shape=(B,C,H,W)
            preds: predictions, shape=(B,num_class)
            targets: targets,   shape=(B,)
        '''
        B, C, H, W = feats.size()
        sp = self.sp

        # L_div (softmax -> CCMP -> summation)
        feature = feats  # (B, C, H, W)
        SF_list = []
        for s, e in zip(sp[:-1], sp[1:]):
            tmp = feature[:, s:e]        # (B, e-s+1, H, W)
            tmp = tmp.view(B, -1, H*W)   # (B, e-s+1, HW)
            tmp = F.softmax(tmp, dim=2)  # (B, e-s+1, HW)
            SF_list.append(tmp)

        L_div = 0.
        for i, feature in enumerate(SF_list):
            feature = F.max_pool2d(feature,
                                   kernel_size=(self.cnums[i], 1),
                                   stride=(self.cnums[i], 1))  # (B, ?, HW)

            Sum = torch.mean(torch.sum(feature, dim=2))
            AvgFactor = self.cnums[i] * 1.0
            L_div = L_div + (1.0 - Sum / AvgFactor)

        # L_dis (CWA -> CCMP -> GAP -> CE)
        mask = self._gen_mask(self.cnums, self.cgroups, self.p)
        if feats.is_cuda:
            mask = mask.cuda()

        feature = feats * mask   # CWA, shape = (B, C, H, W)
        CWA_list = []
        for s, e in zip(sp[:-1], sp[1:]):
            CWA_list.append(feature[:, s:e])  # (B, e-s+1, H, W)

        dis_list = []
        for i, feature in enumerate(CWA_list):
            feature = F.max_pool2d(feature.view(B, -1, H*W),
                                   kernel_size=(self.cnums[i], 1),
                                   stride=(self.cnums[i], 1))  # (B, ?, HW)
            dis_list.append(feature)

        dis_list = torch.cat(dis_list, dim=1).view(B, -1, H, W)  # CCMP
        dis_list = self.avgpool(dis_list).view(B, -1)  # GAP

        L_dis = self.celoss(dis_list, targets)

        # L_CE
        L_CE = self.celoss(preds, targets)

        return L_CE + self.mu_ * (L_dis + self.lambda_ * L_div)

    def _gen_mask(self, cnums, cgroups, p):
        bar = []
        for i in range(len(cnums)):
            foo = np.ones((cgroups[i], cnums[i]), dtype=np.float32).reshape(-1,)
            drop_num = int(cnums[i] * p)
            drop_idx = []
            for j in range(cgroups[i]):
                drop_idx.append(np.random.choice(np.arange(cnums[i]), size=drop_num, replace=False) + j * cnums[i])

            drop_idx = np.stack(drop_idx, axis=0).reshape(-1,)
            foo[drop_idx] = 0.
            bar.append(foo)
        bar = np.hstack(bar).reshape(1, -1, 1, 1)
        bar = torch.from_numpy(bar)

        return bar

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    mcloss = MCCE_Loss()
    targets = torch.from_numpy(np.arange(2)).long()
    feat = torch.randn((2, 1792, 13, 13))
    loss = mcloss(feat, F.one_hot(targets).float(), targets)
    print(loss)
