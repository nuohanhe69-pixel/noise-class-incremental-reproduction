from typing import Tuple, Union

import einops
import torch
import torch.nn as nn

from backbone import MammothBackbone, register_backbone
# from backbone.utils.modules import Pos_Embed


CONFIGS = {
    "tiny": [
        [32, 32, 16], [32, 32, 16],
        [32, 64, 32], [64, 64, 32],
        [64, 128, 64], [128, 128, 64],
        [128, 128, 64], [128, 128, 64],
    ],
    "base": [
        [64, 64, 16], [64, 64, 16],
        [64, 128, 32], [128, 128, 32],
        [128, 256, 64], [256, 256, 64],
        [256, 256, 64], [256, 256, 64],
    ],
}


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class STA_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        qkv_dim,
        num_frames,
        num_joints,
        num_heads,
        kernel_size,
        use_pes=False,
        att_drop=0.0,
    ):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pes
        pads = int((kernel_size[1] - 1) / 2)
        padt = int((kernel_size[0] - 1) / 2)

        if self.use_pes:
            self.pes = Pos_Embed(in_channels, num_frames, num_joints)
        self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.values = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.att_bias = nn.Parameter(torch.ones(1, num_heads, 1, 1) / num_joints, requires_grad=True)
        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)),
            nn.BatchNorm2d(out_channels),
        )
        self.ff_net = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), padding=(padt, 0)),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels:
            self.ress = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            self.rest = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.ress = lambda x: x
            self.rest = lambda x: x

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(att_drop)

    def forward(self, x):
        n, c, t, v = x.size()
        x_spatial = self.pes(x) + x if self.use_pes else x

        q, k = torch.chunk(self.to_qkvs(x_spatial).view(n, 2 * self.num_heads, self.qkv_dim, t, v), 2, dim=1)
        attention = torch.einsum("nhctu,nhctv->nhuv", (q, k)) / (self.qkv_dim * t) * self.values
        attention = attention + self.att_bias.repeat(n, 1, v, v)
        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        x_spatial = torch.einsum("nctu,nhuv->nhctv", (x, attention)).contiguous().view(
            n, self.num_heads * self.in_channels, t, v
        )
        x_res_spatial = self.ress(x)

        x_spatial = self.relu(self.out_nets(x_spatial) + x_res_spatial)
        x_spatial = self.relu(self.ff_net(x_spatial) + x_res_spatial)

        return self.relu(self.out_nett(x_spatial) + self.rest(x_spatial))


@register_backbone("sttformer")
class STTFormer(MammothBackbone):
    def __init__(
        self,
        num_joints: int = 25,
        num_frames: int = 120,
        num_classes: int = 60,
        num_channels: int = 3,
        config: Union[str, list] = "tiny",
        len_parts: int = 6,
        num_heads: int = 3,
        kernel_size: Tuple[int, int] = (3, 5),
        use_pes: bool = False,
        att_drop: float = 0,
        dropout: float = 0,
        dropout2d: float = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        config = CONFIGS.get(config) if config in ("tiny", "base") else config
        self.len_parts = len_parts
        self.in_channels = config[0][0]
        self.out_channels = config[-1][1]
        self.num_classes = num_classes
        self.num_frames = num_frames // self.len_parts
        self.num_joints = num_joints * self.len_parts

        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, self.in_channels, 1),
            nn.BatchNorm2d(self.in_channels),
            nn.LeakyReLU(0.1),
        )
        self.blocks = nn.ModuleList()
        self.pes = Pos_Embed(self.in_channels, self.num_frames, self.num_joints)

        for in_channels, out_channels, qkv_dim in config:
            self.blocks.append(
                STA_Block(
                    in_channels,
                    out_channels,
                    qkv_dim,
                    num_frames=self.num_frames,
                    num_joints=self.num_joints,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    use_pes=use_pes,
                    att_drop=att_drop,
                )
            )

        self.classifier = nn.Linear(self.out_channels, num_classes)
        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)
        self.skel_token = nn.Parameter(torch.zeros(1, self.out_channels), requires_grad=True)
        self.return_features = False

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_init(module)
            elif isinstance(module, nn.BatchNorm2d):
                bn_init(module, 1)
            elif isinstance(module, nn.Linear):
                fc_init(module)

    def forward(self, x, returnt="out"):
        batch_size = x.shape[0]
        two_skel_mask = x[..., 1].sum((1, 2, 3)) > 0

        if not two_skel_mask.any():
            x = x[..., 0]
        else:
            x = torch.cat([x[..., 0], x[two_skel_mask, ..., 1]], dim=0)
        x = einops.rearrange(x, "n c (t p) v -> n c t (p v)", p=self.len_parts)

        x = self.input_map(x)
        x += self.pes()

        for block in self.blocks:
            x = block(x)

        if self.return_features:
            return self.add_second_skel_token(x, batch_size, two_skel_mask)

        x = einops.rearrange(x, "nm c t v -> nm (t v) c 1")
        x = self.drop_out2d(x)
        x = einops.reduce(x, "nm tv c 1 -> nm c", "mean")
        x = self.drop_out(x)
        features = self.add_second_skel_token(x, batch_size, two_skel_mask)

        if returnt == "features":
            return features
        logits = self.classifier(features)
        if returnt == "both":
            return logits, features
        return logits

    def add_second_skel_token(self, x: torch.Tensor, batch_size: int, two_skel_mask: torch.Tensor) -> torch.Tensor:
        second_skel = torch.zeros_like(x[:batch_size])
        if two_skel_mask.any():
            second_skel[two_skel_mask] = x[batch_size:]
        if len(x.shape) == 4:
            second_skel[~two_skel_mask] = self.skel_token.repeat((~two_skel_mask).sum().item(), 1).unsqueeze(-1).unsqueeze(-1)
        else:
            second_skel[~two_skel_mask] = self.skel_token.repeat((~two_skel_mask).sum().item(), 1)
        return x[:batch_size] + second_skel
