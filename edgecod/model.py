"""EdgeCOD: pretrained hierarchical encoder -> 1x1 neck -> compact top-down decoder (+ optional HMoE) -> head.

Section numbers refer to the paper (CMES, 2026).
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDES = (4, 8, 16, 32)


class ConvBN(nn.Sequential):
    """k x k convolution + BatchNorm (+ ReLU)."""

    def __init__(self, cin, cout, k=1, dilation=1, groups=1, act=True, stride=1):
        layers = [nn.Conv2d(cin, cout, k, stride=stride, padding=dilation * (k - 1) // 2, dilation=dilation,
                            groups=groups, bias=False),
                  nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class SepConvBN(nn.Sequential):
    """Depth-wise k x k + point-wise 1x1 (the separable decoder of EdgeCOD-S and EdgeCOD-B)."""

    def __init__(self, cin, cout, k=3, dilation=1, act=True):
        super().__init__(ConvBN(cin, cin, k, dilation, groups=cin),
                         ConvBN(cin, cout, 1, act=act))


class HMoE(nn.Module):
    """Heterogeneous Mixture-of-Experts refinement (Section 3.4).

    Depth-wise experts with different receptive fields (3x3, 3x3 with dilation 2, 5x5, 7x7). A 1x1 router gives
    per-pixel softmax weights g, so every location chooses its own receptive field. The weighted sum of the experts
    is projected by a 1x1 convolution and added to the input (residual).
    """

    def __init__(self, c, experts=((3, 1), (3, 2), (5, 1), (7, 1))):
        super().__init__()
        self.experts = nn.ModuleList(ConvBN(c, c, k, d, groups=c) for k, d in experts)
        self.router = nn.Conv2d(c, len(experts), 1)
        self.proj = ConvBN(c, c, 1, act=False)

    def gates(self, x):
        """Per-pixel routing weights, B x E x H x W (used for the routing analysis, Fig. 5)."""
        return torch.softmax(self.router(x), 1)

    def forward(self, x):
        g = self.gates(x)
        y = sum(g[:, i:i + 1] * e(x) for i, e in enumerate(self.experts))
        return F.relu(x + self.proj(y))


class Backbone(nn.Module):
    """Any timm encoder exposed as four NCHW feature maps at strides 4, 8, 16 and 32."""

    def __init__(self, name, pretrained=True):
        super().__init__()
        probe = timm.create_model(name, pretrained=False, features_only=True)
        reductions = probe.feature_info.reduction()
        missing = [s for s in STRIDES if s not in reductions]
        if missing:
            raise ValueError(f'{name}: no features at strides {missing} (has {reductions})')
        out_indices = tuple(reductions.index(s) for s in STRIDES)
        del probe
        self.body = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=out_indices)
        self.channels = list(self.body.feature_info.channels())

    def forward(self, x):
        return self.body(x)


class Decoder(nn.Module):
    """Top-down decoder (Section 3.3).

    From the deepest level to the shallowest: upsample, concatenate with the skip feature (order: skip, decoder),
    dropout, 3x3 fusion convolution, then HMoE if enabled. The head is a 3x3 convolution and a 1x1 classifier.
    """

    def __init__(self, c, dropout=0.3, separable=False, hmoe=False):
        super().__init__()
        conv = SepConvBN if separable else ConvBN
        self.drop = nn.Dropout(dropout)
        self.fuse = nn.ModuleList(conv(2 * c, c, 3) for _ in range(3))
        self.hmoe = nn.ModuleList(HMoE(c) for _ in range(3)) if hmoe else None
        self.head = nn.Sequential(conv(c, c, 3), nn.Conv2d(c, 1, 1))

    def forward(self, feats):
        d = feats[3]
        for i, f in enumerate(reversed(feats[:3])):
            d = F.interpolate(d, size=f.shape[2:], mode='bilinear', align_corners=False)
            d = self.fuse[i](self.drop(torch.cat([f, d], 1)))
            if self.hmoe is not None:
                d = self.hmoe[i](d)
        return self.head(d)


class EdgeCOD(nn.Module):
    """EdgeCOD. Returns the logit map at the input resolution (apply a sigmoid for probabilities)."""

    def __init__(self, backbone, width=64, separable=False, hmoe=False, dropout=0.3, pretrained=True):
        super().__init__()
        self.backbone = Backbone(backbone, pretrained)
        self.neck = nn.ModuleList(ConvBN(ch, width, 1) for ch in self.backbone.channels)
        self.decoder = Decoder(width, dropout, separable, hmoe)

    @classmethod
    def from_config(cls, mc, pretrained=None):
        return cls(mc['backbone'], mc.get('width', 64), mc.get('separable', False), mc.get('hmoe', False),
                   mc.get('dropout', 0.3), mc.get('pretrained', True) if pretrained is None else pretrained)

    def forward(self, x):
        feats = [neck(f) for f, neck in zip(self.backbone(x), self.neck)]
        return F.interpolate(self.decoder(feats), size=x.shape[2:], mode='bilinear', align_corners=False)


def load_weights(model, path):
    """Load a released checkpoint ({'model': state_dict, ...} or a bare state_dict)."""
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(ck.get('model', ck))
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    bb = sum(p.numel() for p in model.backbone.parameters())
    return {'params_M': total / 1e6, 'backbone_M': bb / 1e6, 'head_M': (total - bb) / 1e6}
