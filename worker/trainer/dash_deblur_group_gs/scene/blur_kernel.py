#
# 版权所有 (C) 2023, Inria
# GRAPHDECO 研究组, https://team.inria.fr/graphdeco
# 初始化输出目录
#
# 本软件仅可在 LICENSE.md 文件条款下用于
# 非商业、研究和评估用途。
#
# 咨询请联系：george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import numpy as np


class Embedder:
    def __init__(self, **kwargs):
        # kwargs 保存编码器超参数，如输入维度、频率数量、是否包含原始输入等。
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        # 构建位置编码函数列表：[..., sin(2^k x), cos(2^k x), ...]
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        # 将所有编码函数输出在最后一个维度拼接。
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    # i == -1 时返回恒等映射，常用于关闭位置编码。
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : i,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class GTnet(nn.Module):
    def __init__(self, res_pos=3, res_view=10, num_hidden=3, width=64, pos_delta=False, num_moments=4):
        super().__init__()
        # pos_delta=False：散焦模糊分支（不预测位置偏移）
        # pos_delta=True ：运动模糊分支（预测多个时刻的位置偏移）
        self.pos_delta = pos_delta
        self.num_moments = num_moments

        self.embed_pos, self.embed_pos_cnl = get_embedder(res_pos, 3)
        self.embed_view, self.embed_view_cnl = get_embedder(res_view, 3)
        in_cnl = self.embed_pos_cnl + self.embed_view_cnl + 7 # 初始化输出目录 7 ??????????

        hiddens = [nn.Linear(width, width) if i % 2 == 0 else nn.ReLU()
                    for i in range((num_hidden - 1) * 2)]

        self.linears = nn.Sequential(
                nn.Linear(in_cnl, width),
                nn.ReLU(),
                *hiddens,
        ).to("cuda")
        if not pos_delta:   # Defocus（散焦模糊）
            self.s = nn.Linear(width, 3).to("cuda")
            self.r = nn.Linear(width, 4).to("cuda")
        else:   # Motion（相机运动模糊）
            self.s = nn.Linear(width, 3*(num_moments + 1)).to("cuda")
            self.r = nn.Linear(width, 4*(num_moments + 1)).to("cuda")
            self.p = nn.Linear(width, 3*num_moments).to("cuda")

        self.linears.apply(init_linear_weights)
        self.s.apply(init_linear_weights)
        self.r.apply(init_linear_weights)
        if pos_delta:
            self.p.apply(init_linear_weights)
            
    def forward(self, pos, scales, rotations, viewdirs):
        """
        输入:
            pos:       [N, 3] 高斯中心（或局部位置）
            scales:    [N, 3] 当前高斯尺度
            rotations: [N, 4] 当前高斯旋转（四元数）
            viewdirs:  [N, 3] 视线方向

        输出:
            scales_delta:    尺度增量
            rotations_delta: 旋转增量
            pos_delta:       位置增量（仅运动模糊分支）
        """
        pos_delta = None
        pos = self.embed_pos(pos)
        viewdirs = self.embed_view(viewdirs)

        x = torch.cat([pos, viewdirs, scales, rotations], dim=-1)
        x1 = self.linears(x)

        scales_delta = self.s(x1)
        rotations_delta = self.r(x1)

        if self.pos_delta:
            pos_delta = self.p(x1)

        return scales_delta, rotations_delta, pos_delta
        
"""
去模糊核网络（GTnet）与位置编码工具。

本文件主要用于 Deblurring 3DGS 训练阶段的“可学习模糊建模”：
1. 对输入位置/视线方向做频率编码（类似 NeRF positional encoding）；
2. 根据当前高斯参数（scale/rotation）预测校正量；
3. 在运动模糊场景下，可额外预测位置偏移分量。
"""
