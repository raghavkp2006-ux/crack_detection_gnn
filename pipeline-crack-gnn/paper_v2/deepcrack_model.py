import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class DeepCrackHED(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        f = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None).features
        self.blocks = nn.ModuleList([f[0:4], f[4:9], f[9:16], f[16:23], f[23:30]])   # conv1..conv5 (last pool dropped)
        self.side = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in (64, 128, 256, 512, 512)])
        self.fuse = nn.Conv2d(5, 1, 1)
        nn.init.constant_(self.fuse.weight, 0.2); nn.init.zeros_(self.fuse.bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
        x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        sides = []
        for blk, sc in zip(self.blocks, self.side):
            x = blk(x)
            sides.append(F.interpolate(sc(x), size=(H + ph, W + pw), mode='bilinear', align_corners=False)[..., :H, :W])
        fused = self.fuse(torch.cat(sides, 1))
        return fused, sides                                    # logits

def cb_bce(logit, y):
    """Class-balanced BCE (HED style): weight each class by the other's frequency in the batch."""
    pos = y.sum(); n = float(y.numel()); w = torch.where(y > 0.5, (n - pos) / n, pos / n)
    return F.binary_cross_entropy_with_logits(logit, y, weight=w)
