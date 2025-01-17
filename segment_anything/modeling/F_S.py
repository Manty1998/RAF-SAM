import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from .common import LayerNorm2d
from einops import rearrange
import math

# 引入 FrequancyEncoding 和 SegField 模块
class FrequancyEncoding(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

    def get_out_dim(self) -> int:
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(self, in_tensor):
        # 使用 torch 的 pi 常数和计算逻辑
        scaled_in_tensor = 2 * math.pi * in_tensor  # 将值缩放到 [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        # 生成编码的输入
        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + math.pi / 2.0], dim=-1))

        # 如果包含输入
        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs




class SegField(nn.Module):
    def __init__(self,
                 features: int = 128,
                 cls_channel: int = 1,
                 coord_normalized_min: float = -1.0,
                 coord_normalized_max: float = 1.0,
                 coords_channel: int = 2,  # ACDC 数据集主要关注 2D MRI 图像，因此将坐标维度调整为 2
                 w0: int = 10,
                 top_k_ratio: float = 0.2):  # top_k_ratio 增大来确保细节更好地保留
        super().__init__()

        self.coord_normalized_min = coord_normalized_min
        self.coord_normalized_max = coord_normalized_max
        self.cls_channel = cls_channel
        self.out_channel = 1
        self.pos_enc = FrequancyEncoding(in_dim=coords_channel, num_frequencies=w0, min_freq_exp=0.0, max_freq_exp=w0,
                                         include_input=True)

        self.top_k_ratio = top_k_ratio

        # 根据 ACDC 数据集特点调整输入维度
        input_dim = 32 + 128 + self.pos_enc.get_out_dim()
        fine_input_dim = features + 32 + 128 + self.pos_enc.get_out_dim()

        # 定义分割网络
        self.seg_net = nn.ModuleList()
        self.seg_net.append(
            nn.Sequential(
                nn.Linear(input_dim, features * 4), nn.BatchNorm1d(features * 4), nn.ReLU(),
                nn.Linear(features * 4, features * 2), nn.BatchNorm1d(features * 2), nn.ReLU(),
            )
        )

        self.drop_out = nn.Dropout(0.5)
        self.seg_net.append(
            nn.Sequential(
                nn.Linear(fine_input_dim, features), nn.BatchNorm1d(features), nn.ReLU(),
                nn.Linear(features, features), nn.BatchNorm1d(features), nn.ReLU(),
            )
        )
        self.seg_net.append(
            nn.ModuleList(
                [nn.Sequential(nn.Linear(features * 2, features + self.out_channel)) for _ in range(self.cls_channel)])
        )
        self.seg_net.append(
            nn.ModuleList([nn.Sequential(nn.Linear(features, features // 2), nn.BatchNorm1d(features // 2), nn.ReLU(),
                                         nn.Linear(features // 2, self.out_channel)) for _ in range(self.cls_channel)])
        )
        for net in self.seg_net:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, image_embedding, image_pe, original_shape, epoch_T=1.0, cls=0):
        # 处理 ACDC 数据集的输入
        b, _, h, w = image_embedding.shape
        h, w = original_shape

        d = 1
        coordinates = torch.stack(
            torch.meshgrid(
                torch.linspace(self.coord_normalized_min, self.coord_normalized_max, h),
                torch.linspace(self.coord_normalized_min, self.coord_normalized_max, w),

            ),
            dim=-1,
        ).to(image_embedding.device)

        # 对每个 batch 重复坐标信息
        coordinates = coordinates.unsqueeze(0).repeat(image_embedding.shape[0], 1, 1, 1)
        coordinates = self.pos_enc(coordinates)

        coordinates = rearrange(coordinates, 'b h w c -> (b h w) c')
        mask_coordinates = min(int(epoch_T * (coordinates.shape[-1])) + 3, coordinates.shape[-1])
        coordinates[:, mask_coordinates:] = 0

        # 调整 image_embedding 的维度
        image_embedding = F.interpolate(image_embedding, size=original_shape, mode='bilinear', align_corners=False)

        # 确保 image_pe 的维度与插值需要的维度一致
        if image_pe.dim() == 3:
            # 增加一个维度以匹配插值的输入要求
            image_pe = image_pe.unsqueeze(1)

        image_pe = F.interpolate(image_pe, size=(h, w), mode='bilinear', align_corners=False)

        # 将数据拉平
        image_embedding_flatten = rearrange(image_embedding, 'b c h w -> (b h w) c')
        image_pe_flatten = rearrange(image_pe, 'b c h w -> (b h w) c')

        # 拼接特征和位置编码
        feat_concat = torch.cat([image_embedding_flatten, image_pe_flatten, coordinates], dim=1)
        seg_res = []

        seg_feat_concat = []
        for _ in range(2):
            seg_feat_ = self.seg_net[0](feat_concat)
            seg_feat_ = self.seg_net[2][cls](seg_feat_)
            seg_feat_ = self.drop_out(seg_feat_)
            seg_feat_concat.append(seg_feat_)
        seg_feat_concat = torch.stack(seg_feat_concat, dim=1)

        seg_feat = seg_feat_
        seg_feat_var = torch.var(seg_feat_concat, dim=1).mean(dim=1, keepdim=True)

        seg_coarse = seg_feat[:, 0:self.out_channel]
        seg_fine = seg_coarse.clone()
        seg_feat = seg_feat[:, self.out_channel:]
        seg_coarse = rearrange(seg_coarse, '(b h w) c -> b c h w', b=b, h=h, w=w)
        seg_res.append(seg_coarse)

        # 细化阶段，根据方差选择最重要的像素
        _, sel_ind = torch.topk(seg_feat_var[:, 0], k=int(seg_feat_var.shape[0] * self.top_k_ratio))
        fine_feat_concat = torch.cat([feat_concat[sel_ind, ...], seg_feat[sel_ind, ...]], dim=1)
        seg_fine_sel = self.seg_net[1](fine_feat_concat)
        seg_fine[sel_ind, ...] = self.seg_net[3][cls](seg_fine_sel)[:, 0:self.out_channel]

        seg_fine = rearrange(seg_fine, '(b h w) c -> b c h w', b=b, h=h, w=w)
        seg_res.append(seg_fine)

        return seg_res



