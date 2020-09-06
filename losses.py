"""
Dice loss 3D
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def dice_score(preds, targets):
    num = 2*torch.einsum('bcijk, bcijk ->bc', [preds, targets])
    denom = torch.einsum('bcijk, bcijk -> bc', [preds, preds]) +\
        torch.einsum('bcijk, bcijk -> bc', [targets, targets]) + 1e-8
    proportions = torch.div(num, denom) 
    return torch.einsum('bc->c', proportions)


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mu, logvar, N):
        sum_square_mean = torch.einsum('i,i->', mu, mu)
        sum_log_var = torch.einsum('i->', logvar)
        sum_var = torch.einsum('i->', torch.exp(logvar))
        print(f'ssm: {sum_square_mean}\tslv: {sum_log_var}\t svr: {sum_var}\t') 
        return float(1/N)*(sum_square_mean+sum_var-sum_log_var-N)


class VAEDiceLoss(nn.Module):
    def __init__(self):
        super(VAEDiceLoss, self).__init__()
        self.avgdice = AvgDiceLoss()
        self.kl = KLLoss()

    def forward(self, output, targets):
        ad = self.avgdice(output['seg_map'], targets)
        ms = 0.1*F.mse_loss(output['recon'], targets['src'])
        kl = 0.1*self.kl(output['mu'], output['logvar'], 256)

        print(f'ad: {ad}\tms: {ms}\tkl: {kl}')
        return ad + ms + kl
        #return self.avgdice(output['seg_map'], targets)\
        #    + 0.1*F.mse_loss(output['recon'], targets['src'])\
        #    + 0.1*self.kl(output['mu'], output['logvar'], 256)

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice = AvgDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, logits, targets):
        target = targets['target']
        return self.dice(preds, targets) + self.bce(logits[:, 0, :, :, :].squeeze(), target[:, 0, :, :, :].squeeze())\
                + self.bce(logits[:, 1, :, :, :].squeeze(), target[:, 1, :, :, :].squeeze())\
                + self.bce(logits[:, 2, :, :, :].squeeze(), target[:, 2, :, :, :].squeeze())


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, logits, targets):
        target = targets['target']
        return self.bce(logits[:, 0, :, :, :].squeeze(), target[:, 0, :, :, :].squeeze())\
                + self.bce(logits[:, 1, :, :, :].squeeze(), target[:, 1, :, :, :].squeeze())\
                + self.bce(logits[:, 2, :, :, :].squeeze(), target[:, 2, :, :, :].squeeze())
        

class AvgDiceLoss(nn.Module):
    def __init__(self):
        super(AvgDiceLoss, self).__init__()
    
    def forward(self, preds, targets):
        # this conditional is used for CascadeNet when only using bilinear upsampling
        if preds == None:
            return 0
        target = targets['target']
        proportions = dice_score(preds, target)
        avg_dice = torch.einsum('c->', proportions) / (target.shape[0]*target.shape[1])
        return 1 - avg_dice


class WTLoss(nn.Module):
    def __init__(self):
        super(WTLoss, self).__init__()
    
    def forward(self, preds, targets):
        target = targets['target']
        wt_preds = preds.new_zeros(preds.size()) 
        wt_targets = target.new_zeros(target.size()) 
        wt_preds[:, 0, :, :, :] = wt_preds[:, 1, :, :, :] = wt_preds[:, 2, :, :, :] = preds[:, 1, :, :, :]
        wt_targets[:, 0, :, :, :] = wt_targets[:, 1, :, :, :] = wt_targets[:, 2, :, :, :] = target[:, 1, :, :, :]
        proportions = dice_score(wt_preds, wt_targets)
        wt = torch.einsum('c->', proportions) / (target.shape[0]*target.shape[1])
        return 1 - wt


class CascadeAvgDiceLoss(nn.Module):
    def __init__(self, coarse_wt_only=False):
        super(CascadeAvgDiceLoss, self).__init__()
        if coarse_wt_only:
            self.coarse_loss = WTLoss()
        else:
            self.coarse_loss = AvgDiceLoss()
        self.deconv_loss = AvgDiceLoss()
        self.biline_loss = AvgDiceLoss()

    def forward(self, output, targets):
        return self.coarse_loss(output['coarse'], targets)\
                + self.biline_loss(output['biline'], targets)\
                + self.deconv_loss(output['deconv'], targets)


class AvgDiceEnhanceLoss(nn.Module):
    def __init__(self, device):
        super(AvgDiceEnhanceLoss, self).__init__()
        self.device = device
        self.avgdice = AvgDiceLoss()
    
    def _enhancing_loss(self, et_prob, ce_ratio):
        ''' penalizes et predicted for voxels which are not enhancing 
        by computing the false positive rate of enhancing tumor predictions
        to voxels which are enhancing.'''

    
        # is this a reference? does it change the original tensor?
        # turns out it did and it was a problem
        ce_ratio_ones = torch.zeros(ce_ratio.size()).to(self.device)
        ce_ratio_ones[ce_ratio<=0] = 0 
        #et_pred = torch.zeros(et_prob.size()).to(self.device)
        #et_pred[et_prob>0] = 1
        #fp = torch.einsum('bijk, bijk->b', [et_pred, et_pred])\
        #       - torch.einsum('bijk, bijk->b', [et_pred, ce_ratio])
        fp = torch.einsum('bijk, bijk->b', [et_prob, et_prob])\
               - torch.einsum('bijk, bijk->b', [et_prob, ce_ratio_ones])
        # (this is probably a convoluted way to) compute the number of true negatives
        #tn_mat = torch.ones(et_pred.size()).to(self.device) 
        #tn_mat[et_pred == 0] = 0
        tn_mat = torch.ones(et_prob.size()).to(self.device) 
        tn_mat[et_prob == 0] = 0
        tn_mat[ce_ratio == 0] = 0
        tn_num = torch.einsum('bijk, bijk->b', [tn_mat, tn_mat])

        # false postive ratio
        fp_rat = fp / (fp + tn_num + 1e-32)
        return fp_rat

 
    def forward(self, preds, targets):
        # preds dim BxCxHxWxD
        ad_loss = self.avgdice(preds, targets)
        enh_loss = self._enhancing_loss(
                torch.squeeze(preds[:, 0, :, :, :]), 
                torch.squeeze(targets['src'][:, -1, :, :, :])
                )
        enh_loss_batch_avg = torch.einsum('b->', enh_loss) / len(enh_loss)
        return 0.9*ad_loss + 0.1*enh_loss_batch_avg


def agg_dice_score(preds, targets):
    ''' Gives Dice score for sub-regions which are evaluated in the
    competition.
    '''
    channel_shape = preds[:, 0, :, :, :].size()

    agg = torch.zeros(preds.size())
    et = torch.zeros(channel_shape)
    et[torch.where(preds[:, 2, :, :, :] > 0.5)] = 1
    et = et.unsqueeze(1)

    wt = torch.zeros(channel_shape)
    wt[torch.where((preds[:, 0, :, :, :] > 0.5) | 
    (preds[:, 1, :, :, :] > 0.5) | 
    (preds[:, 2, :, :, :] > 0.5) )] = 1
    wt = wt.unsqueeze(1)

    tc = torch.zeros(channel_shape)
    tc[torch.where((preds[:, 0, :, :, :] > 0.5) | (preds[:, 2, :, :, :] > 0.5) )] = 1
    tc = tc.unsqueeze(1)

    agg_preds = torch.cat((et, wt, tc), 1)

    et_target = torch.zeros(channel_shape)
    et_target[torch.where(targets[:, 2, :, :, :] > 0.5)] = 1
    et_target = et_target.unsqueeze(1) 

    wt_target = torch.zeros(channel_shape)
    wt_target[torch.where((targets[:, 0, :, :, :] > 0.5) | 
    (targets[:, 1, :, :, :] > 0.5) | (targets[:, 2, :, :, :] > 0.5) )] = 1
    wt_target = wt_target.unsqueeze(1)

    tc_target = torch.zeros(channel_shape)
    tc_target[torch.where((targets[:, 0, :, :, :] > 0.5) | 
    (targets[:, 2, :, :, :] > 0.5) )] = 1
    tc_target = tc_target.unsqueeze(1)

    agg_targets = torch.cat((et_target, wt_target, tc_target), 1)

    return dice_score(agg_preds, agg_targets)



