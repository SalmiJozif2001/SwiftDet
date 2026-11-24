import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBNReLU, QARepConv2d, PConvBlock, SPPFLight


class SwiftDetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem_1 = ConvBNReLU(3, 16, k=3, s=2, p=1, act=True)
        self.stem_2 = QARepConv2d(16, 32, k=3, s=2, p=1)
        self.stage1 = nn.Sequential(PConvBlock(32, ratio=0.25, mlp_ratio=2.0))

        self.down2 = QARepConv2d(32, 64, k=3, s=2, p=1)
        self.stage2 = nn.Sequential(*[PConvBlock(64, 0.25, 2.0) for _ in range(2)])

        self.down3 = QARepConv2d(64, 128, k=3, s=2, p=1)
        self.stage3 = nn.Sequential(*[PConvBlock(128, 0.25, 2.0) for _ in range(3)])

        self.down4 = QARepConv2d(128, 256, k=3, s=2, p=1)
        self.stage4 = nn.Sequential(*[PConvBlock(256, 0.25, 2.0) for _ in range(2)])
        self.sppf = SPPFLight(256, k=5)

    def forward(self, x):
        x = self.stage1(self.stem_2(self.stem_1(x)))
        c3 = self.stage2(self.down2(x))
        c4 = self.stage3(self.down3(c3))
        c5 = self.sppf(self.stage4(self.down4(c4)))
        return c3, c4, c5


class SlimPANPConv(nn.Module):
    def __init__(self, in_channels=(64, 128, 256), cn=64):
        super().__init__()
        c3, c4, c5 = in_channels
        self.lat_p5 = ConvBNReLU(c5, cn, k=1, s=1, p=0)
        self.lat_p4 = ConvBNReLU(c4, cn, k=1, s=1, p=0)
        self.lat_p3 = ConvBNReLU(c3, cn, k=1, s=1, p=0)

        self.td_p4_red = ConvBNReLU(2 * cn, cn, k=1, s=1, p=0)
        self.td_p4 = PConvBlock(cn, ratio=0.25, mlp_ratio=2.0)

        self.td_p3_red = ConvBNReLU(2 * cn, cn, k=1, s=1, p=0)
        self.td_p3 = PConvBlock(cn, ratio=0.25, mlp_ratio=2.0)

        self.bu_p3_down = ConvBNReLU(cn, cn, k=3, s=2, p=1)
        self.bu_p4_red = ConvBNReLU(2 * cn, cn, k=1, s=1, p=0)
        self.bu_p4 = PConvBlock(cn, ratio=0.25, mlp_ratio=2.0)

        self.bu_p4_down = ConvBNReLU(cn, cn, k=3, s=2, p=1)
        self.bu_p5_red = ConvBNReLU(2 * cn, cn, k=1, s=1, p=0)
        self.bu_p5 = PConvBlock(cn, ratio=0.25, mlp_ratio=2.0)

    def forward(self, feats):
        c3, c4, c5 = feats
        l5 = self.lat_p5(c5)
        l4 = self.lat_p4(c4)
        l3 = self.lat_p3(c3)

        up5 = F.interpolate(l5, size=l4.shape[-2:], mode="nearest")
        p4_td = self.td_p4(self.td_p4_red(torch.cat([up5, l4], dim=1)))

        up4 = F.interpolate(p4_td, size=l3.shape[-2:], mode="nearest")
        p3_out = self.td_p3(self.td_p3_red(torch.cat([up4, l3], dim=1)))

        p4_out = self.bu_p4(self.bu_p4_red(torch.cat([self.bu_p3_down(p3_out), p4_td], dim=1)))
        p5_out = self.bu_p5(self.bu_p5_red(torch.cat([self.bu_p4_down(p4_out), l5], dim=1)))
        return p3_out, p4_out, p5_out


class HeadBranch(nn.Module):
    def __init__(self, cn=64, out_ch=20):
        super().__init__()
        self.cp = cn // 2
        self.pconv = nn.Conv2d(self.cp, self.cp, 3, stride=1, padding=1, bias=False)
        self.pw = ConvBNReLU(cn, cn, k=1, s=1, p=0, act=True)
        self.out_conv = nn.Conv2d(cn, out_ch, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.cp, x.shape[1] - self.cp], dim=1)
        feat = self.pw(torch.cat((self.pconv(x1), x2), dim=1))
        return self.out_conv(feat)


class DecoupledGFLHead(nn.Module):
    def __init__(self, cn=64, num_classes=20, reg_max=7):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.cls_branches = nn.ModuleList([HeadBranch(cn, num_classes) for _ in range(3)])
        self.reg_branches = nn.ModuleList([HeadBranch(cn, 4 * (reg_max + 1)) for _ in range(3)])
        self._init_bias()

    def _init_bias(self):
        prior_prob = 0.01
        bias_val = -float(torch.log(torch.tensor((1.0 - prior_prob) / prior_prob)))
        for b in self.cls_branches:
            nn.init.constant_(b.out_conv.bias, bias_val)

    def forward(self, feats):
        outputs = []
        for idx, x in enumerate(feats):
            cls_logit = self.cls_branches[idx](x)
            reg_dist = self.reg_branches[idx](x)
            b, _, h, w = cls_logit.shape
            pred = torch.cat([cls_logit, reg_dist], dim=1)
            pred = pred.permute(0, 2, 3, 1).reshape(b, h * w, self.num_classes + 4 * (self.reg_max + 1))
            outputs.append(pred)
        return torch.cat(outputs, dim=1)


class SwiftDet(nn.Module):
    def __init__(self, num_classes=20, reg_max=7, neck_channels=64):
        super().__init__()
        self.backbone = SwiftDetBackbone()
        self.neck = SlimPANPConv((64, 128, 256), cn=neck_channels)
        self.head = DecoupledGFLHead(cn=neck_channels, num_classes=num_classes, reg_max=reg_max)

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))

    def deploy(self):
        for m in self.modules():
            if hasattr(m, "fuse_for_deploy"):
                m.fuse_for_deploy()
        return self
