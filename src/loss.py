import torch
import torch.nn as nn

class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

def uncertainty_aware_loss(pred, target, uncertainty):
    loss = 0.5 * torch.exp(-uncertainty) * (pred - target)**2 + 0.5 * uncertainty
    return loss.mean()