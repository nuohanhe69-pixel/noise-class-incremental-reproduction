from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import MammothBackbone, register_backbone


def _ntu_edges() -> Tuple[Iterable[Tuple[int, int]], Iterable[Tuple[int, int]]]:
    # NTU RGB+D 25-joint graph. Pairs are (child, parent), zero-indexed.
    inward = [
        (0, 1), (1, 20), (2, 20), (3, 2), (4, 20),
        (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
        (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
        (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
        (21, 22), (22, 7), (23, 24), (24, 11),
    ]
    outward = [(parent, child) for child, parent in inward]
    return inward, outward


def _normalize_digraph(adjacency: torch.Tensor) -> torch.Tensor:
    degree = adjacency.sum(0)
    inv_degree = torch.zeros_like(degree)
    inv_degree[degree > 0] = degree[degree > 0].pow(-1)
    return adjacency @ torch.diag(inv_degree)


def build_ntu_adjacency(num_joints: int = 25) -> torch.Tensor:
    if num_joints != 25:
        raise ValueError("EfficientGCN currently supports the NTU 25-joint layout.")

    adjacency = torch.zeros(3, num_joints, num_joints)
    adjacency[0] = torch.eye(num_joints)
    inward, outward = _ntu_edges()

    for child, parent in inward:
        adjacency[1, child, parent] = 1
    for parent, child in outward:
        adjacency[2, parent, child] = 1

    adjacency[1] = _normalize_digraph(adjacency[1])
    adjacency[2] = _normalize_digraph(adjacency[2])
    return adjacency


def bn_init(module: nn.BatchNorm2d, scale: float = 1.0) -> None:
    nn.init.constant_(module.weight, scale)
    nn.init.constant_(module.bias, 0)


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor, adaptive: bool = True):
        super().__init__()
        self.num_subsets = adjacency.shape[0]
        self.register_buffer("adjacency", adjacency)
        self.adaptive = adaptive
        if adaptive:
            self.pa = nn.Parameter(torch.zeros_like(adjacency))
        else:
            self.register_parameter("pa", None)

        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subsets, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        adjacency = self.adjacency + self.pa if self.pa is not None else self.adjacency
        x = self.conv(x).view(n, self.num_subsets, -1, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, adjacency)
        return self.bn(x)


class EfficientGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int = 1,
        temporal_kernel: int = 5,
        adaptive: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        padding = (temporal_kernel - 1) // 2
        self.gcn = GraphConv(in_channels, out_channels, adjacency, adaptive=adaptive)
        self.tcn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class InputBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor, adaptive: bool):
        super().__init__()
        hidden = max(out_channels // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            EfficientGCNBlock(hidden, out_channels, adjacency, adaptive=adaptive, residual=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EfficientGCN(MammothBackbone):
    """
    Mammoth-compatible EfficientGCN-B0-style backbone for NTU skeleton streams.

    This implementation keeps the B0 spirit needed by the CL baseline: multiple
    skeleton input branches, NTU graph topology, adaptive graph convolution, and
    depthwise separable temporal convolutions.
    """

    bone_pairs = tuple(_ntu_edges()[0])

    def __init__(
        self,
        num_classes: int = 60,
        num_channels: int = 3,
        num_joints: int = 25,
        num_frames: int = 120,
        num_skeletons: int = 2,
        width: int = 64,
        dropout: float = 0.25,
        adaptive: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.num_skeletons = num_skeletons

        adjacency = build_ntu_adjacency(num_joints)
        self.register_buffer("adjacency", adjacency)

        self.joint_branch = InputBranch(num_channels, width, adjacency, adaptive=adaptive)
        self.bone_branch = InputBranch(num_channels, width, adjacency, adaptive=adaptive)
        self.motion_branch = InputBranch(num_channels, width, adjacency, adaptive=adaptive)

        self.fuse = nn.Sequential(
            nn.Conv2d(width * 3, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        self.layers = nn.Sequential(
            EfficientGCNBlock(width, width, adjacency, adaptive=adaptive),
            EfficientGCNBlock(width, width, adjacency, adaptive=adaptive),
            EfficientGCNBlock(width, width * 2, adjacency, stride=2, adaptive=adaptive),
            EfficientGCNBlock(width * 2, width * 2, adjacency, adaptive=adaptive),
            EfficientGCNBlock(width * 2, width * 4, adjacency, stride=2, adaptive=adaptive),
            EfficientGCNBlock(width * 4, width * 4, adjacency, adaptive=adaptive),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(width * 4, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                bn_init(module)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def _bone_stream(self, x: torch.Tensor) -> torch.Tensor:
        bone = torch.zeros_like(x)
        for child, parent in self.bone_pairs:
            bone[:, :, :, child, :] = x[:, :, :, child, :] - x[:, :, :, parent, :]
        return bone

    @staticmethod
    def _motion_stream(x: torch.Tensor) -> torch.Tensor:
        motion = torch.zeros_like(x)
        motion[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
        return motion

    @staticmethod
    def _merge_persons(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if x.dim() == 5:
            n, c, t, v, m = x.shape
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(n * m, c, t, v)
            return x, n, m
        n, _, _, _ = x.shape
        return x, n, 1

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        joint = x
        bone = self._bone_stream(x)
        motion = self._motion_stream(x)

        joint, batch_size, num_persons = self._merge_persons(joint)
        bone, _, _ = self._merge_persons(bone)
        motion, _, _ = self._merge_persons(motion)

        x = torch.cat(
            [self.joint_branch(joint), self.bone_branch(bone), self.motion_branch(motion)],
            dim=1,
        )
        x = self.fuse(x)
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).flatten(1)
        x = x.view(batch_size, num_persons, -1).mean(dim=1)
        return self.dropout(x)

    def forward(self, x: torch.Tensor, returnt: str = "out") -> torch.Tensor:
        features = self._forward_features(x)
        if returnt == "features":
            return features

        logits = self.fc(features)
        if returnt == "both":
            return logits, features
        if returnt == "all":
            return {"out": logits, "features": features}
        return logits


@register_backbone("efficient-gcn")
def efficient_gcn(
    num_classes: int = 60,
    num_channels: int = 3,
    num_joints: int = 25,
    num_frames: int = 120,
    num_skeletons: int = 2,
    width: int = 64,
    dropout: float = 0.25,
    adaptive: bool = True,
):
    return EfficientGCN(
        num_classes=num_classes,
        num_channels=num_channels,
        num_joints=num_joints,
        num_frames=num_frames,
        num_skeletons=num_skeletons,
        width=width,
        dropout=dropout,
        adaptive=adaptive,
    )
