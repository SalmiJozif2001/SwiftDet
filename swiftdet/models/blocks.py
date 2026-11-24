import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_conv_bn_weights(conv_w, conv_b, bn):
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / std
    w_fused = conv_w * scale.reshape(-1, 1, 1, 1)
    b_base = conv_b if conv_b is not None else torch.zeros_like(bn.running_mean)
    b_fused = bn.bias + (b_base - bn.running_mean) * scale
    return w_fused, b_fused


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        self.deploy_mode = False

    def forward(self, x):
        if self.deploy_mode:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def fuse_for_deploy(self):
        if self.deploy_mode:
            return
        w, b = fuse_conv_bn_weights(self.conv.weight, None, self.bn)
        fused = nn.Conv2d(
            self.conv.in_channels,
            self.conv.out_channels,
            self.conv.kernel_size,
            self.conv.stride,
            self.conv.padding,
            bias=True,
        )
        fused.weight.data.copy_(w)
        fused.bias.data.copy_(b)
        self.conv = fused
        delattr(self, "bn")
        self.deploy_mode = True


class QARepConv2d(nn.Module):
    """Quantization-Aware RepVGG block (Chu et al., AAAI 2024)."""
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        assert k == 3 and p == 1
        self.cin = cin
        self.cout = cout
        self.s = s
        self.deploy_mode = False
        self.act = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(cin, cout, 3, stride=s, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.conv1 = nn.Conv2d(cin, cout, 1, stride=s, padding=0, bias=True)
        self.has_identity = (cin == cout and s == 1)
        self.post_bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        if self.deploy_mode:
            return self.act(self.fused_conv(x))
        out = self.bn3(self.conv3(x)) + self.conv1(x)
        if self.has_identity:
            out = out + x
        return self.act(self.post_bn(out))

    def fuse_for_deploy(self):
        if self.deploy_mode:
            return
        w3, b3 = fuse_conv_bn_weights(self.conv3.weight, None, self.bn3)
        w1 = F.pad(self.conv1.weight, [1, 1, 1, 1])
        b1 = self.conv1.bias
        w_sum = w3 + w1
        b_sum = b3 + b1

        if self.has_identity:
            w_id = torch.zeros_like(w_sum)
            for i in range(self.cin):
                w_id[i, i, 1, 1] = 1.0
            w_sum = w_sum + w_id

        w_final, b_final = fuse_conv_bn_weights(w_sum, b_sum, self.post_bn)
        self.fused_conv = nn.Conv2d(self.cin, self.cout, 3, stride=self.s, padding=1, bias=True)
        self.fused_conv.weight.data.copy_(w_final)
        self.fused_conv.bias.data.copy_(b_final)

        for attr in ("conv3", "bn3", "conv1", "post_bn"):
            delattr(self, attr)
        self.deploy_mode = True


class PConvBlock(nn.Module):
    """FasterNet block: PConv(c/4) -> PW1(c->2c)+BN+ReLU -> PW2(2c->c)+BN + Skip."""
    def __init__(self, c, ratio=0.25, mlp_ratio=2.0):
        super().__init__()
        self.cp = int(c * ratio)
        self.c_rem = c - self.cp
        c_mid = int(c * mlp_ratio)

        self.pconv = nn.Conv2d(self.cp, self.cp, 3, stride=1, padding=1, bias=False)
        self.pw1 = ConvBNReLU(c, c_mid, k=1, s=1, p=0, act=True)
        self.pw2 = ConvBNReLU(c_mid, c, k=1, s=1, p=0, act=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.cp, self.c_rem], dim=1)
        x1 = self.pconv(x1)
        out = torch.cat((x1, x2), dim=1)
        out = self.pw2(self.pw1(out))
        return x + out


class SPPFLight(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        c_ = c // 2
        self.cv1 = ConvBNReLU(c, c_, k=1, s=1, p=0, act=True)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = ConvBNReLU(c_ * 4, c, k=1, s=1, p=0, act=True)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))
