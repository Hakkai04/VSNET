"""
EdgeGuidedSwinUNETR: 双分支边缘引导注意力 Swin-UNETR
============================================================
核心思想：
  1. 编码器：完整复用 MONAI SwinUNETR 的 Swin Transformer 编码器（保留预训练权重）
  2. 主分割解码器：复用 SwinUNETR 的 CNN 解码器，负责体积级血管分割
  3. 边缘分支：轻量级的 3D 卷积头，从编码器浅层特征中提取管壁边界
  4. 边缘引导注意力门 (EAG)：在每一级跳跃连接处，用边缘分支的预测
     作为空间注意力权重，强化管壁边缘特征、抑制背景噪声

输出：
  训练时返回 (seg_logits, edge_logits)
  推理时返回 seg_logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR


class ConvBlock3D(nn.Module):
    """基础的 3D 卷积块：Conv -> InstanceNorm -> LeakyReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EdgeAttentionGate(nn.Module):
    """
    边缘引导注意力门 (Edge-Guided Attention Gate)
    -----------------------------------------------
    输入：
      - x_skip: 来自编码器的跳跃连接特征 (高分辨率，丰富的空间细节)
      - x_up:   来自解码器上采样后的特征 (低分辨率，丰富的语义信息)
      - edge_feat: 来自边缘分支的边界特征图 (与 x_skip 同分辨率)
    
    输出：
      - 被边缘注意力增强后的跳跃连接特征

    机制：
      1. 用 x_up 作为语义"指挥棒"（传统 Attention Gate 思路）
      2. 额外加入 edge_feat 作为第三路输入（边缘先验引导）
      3. 三路信息融合后生成空间注意力权重 α ∈ [0, 1]
      4. 最终输出 = x_skip * (1 + α)，残差连接确保主体特征不丢失
    """
    def __init__(self, F_skip, F_up, F_edge, F_int):
        super().__init__()
        # 将三路特征统一映射到 F_int 维度
        self.W_skip = nn.Sequential(
            nn.Conv3d(F_skip, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(F_int),
        )
        self.W_up = nn.Sequential(
            nn.Conv3d(F_up, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(F_int),
        )
        self.W_edge = nn.Sequential(
            nn.Conv3d(F_edge, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(F_int),
        )
        # 生成注意力权重
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x_skip, x_up, edge_feat):
        # 对齐空间尺寸（x_up 可能与 x_skip 尺寸不完全匹配）
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='trilinear', align_corners=False)
        if edge_feat.shape[2:] != x_skip.shape[2:]:
            edge_feat = F.interpolate(edge_feat, size=x_skip.shape[2:], mode='trilinear', align_corners=False)

        # 三路融合
        g1 = self.W_skip(x_skip)
        g2 = self.W_up(x_up)
        g3 = self.W_edge(edge_feat)
        
        # 加法融合 + 注意力计算
        alpha = self.psi(self.relu(g1 + g2 + g3))  # (N, 1, D, H, W)
        
        # 残差注意力增强：(1 + α) 确保原始信息不被完全截断
        return x_skip * (1.0 + alpha)


class EdgeBranch(nn.Module):
    """
    轻量级边缘检测分支
    -----------------------------------------------
    从编码器的浅层（1/2 分辨率）和中层（1/4 分辨率）特征中，
    提取边界信息并逐级上采样回原始分辨率。
    
    输出：
      - edge_logits: 边缘预测 logits (N, out_channels, D, H, W)
      - edge_features: 多尺度边缘特征字典，供 EAG 使用
    """
    def __init__(self, enc_channels, out_channels=3):
        """
        Args:
            enc_channels: list of encoder feature channels at each resolution
                          e.g. [48, 48, 96, 192, 384, 768] for feature_size=48
            out_channels: 边缘预测的类别数（与分割任务一致）
        """
        super().__init__()
        # 从编码器的不同层级提取边缘特征
        # enc_channels[0] = 48 (原始分辨率映射)
        # enc_channels[1] = 48 (1/2 分辨率)
        # enc_channels[2] = 96 (1/4 分辨率)
        
        edge_hidden = 32  # 边缘分支的隐层宽度（极轻量）
        
        # 从 1/4 分辨率的编码器特征提取边缘
        self.edge_conv_deep = ConvBlock3D(enc_channels[2], edge_hidden)
        # 从 1/2 分辨率的编码器特征提取边缘
        self.edge_conv_shallow = ConvBlock3D(enc_channels[1], edge_hidden)
        # 从原始分辨率特征提取边缘
        self.edge_conv_full = ConvBlock3D(enc_channels[0], edge_hidden)
        
        # 融合多尺度边缘特征
        self.fuse = nn.Sequential(
            ConvBlock3D(edge_hidden * 3, edge_hidden),
            ConvBlock3D(edge_hidden, edge_hidden),
        )
        
        # 最终边缘预测头
        self.edge_head = nn.Conv3d(edge_hidden, out_channels, kernel_size=1)
        
        # 供 EAG 使用的多尺度边缘特征投影
        self.edge_proj_s0 = ConvBlock3D(edge_hidden, edge_hidden, kernel_size=1, padding=0)
        self.edge_proj_s1 = ConvBlock3D(edge_hidden, edge_hidden, kernel_size=1, padding=0)
        self.edge_proj_s2 = ConvBlock3D(edge_hidden, edge_hidden, kernel_size=1, padding=0)
        self.edge_proj_s3 = ConvBlock3D(edge_hidden, edge_hidden, kernel_size=1, padding=0)

    def forward(self, enc_feats, target_size):
        """
        Args:
            enc_feats: list of encoder features [feat_s0, feat_s1, feat_s2, ...]
            target_size: 原始输入图像的空间尺寸 (D, H, W)
        """
        # 提取多尺度边缘特征
        e_deep = self.edge_conv_deep(enc_feats[2])     # 1/4 res
        e_shallow = self.edge_conv_shallow(enc_feats[1]) # 1/2 res
        e_full = self.edge_conv_full(enc_feats[0])       # full res
        
        # 上采样到原始分辨率并拼接
        e_deep_up = F.interpolate(e_deep, size=target_size, mode='trilinear', align_corners=False)
        e_shallow_up = F.interpolate(e_shallow, size=target_size, mode='trilinear', align_corners=False)
        e_full_up = F.interpolate(e_full, size=target_size, mode='trilinear', align_corners=False)
        
        # 多尺度融合
        e_cat = torch.cat([e_full_up, e_shallow_up, e_deep_up], dim=1)
        e_fused = self.fuse(e_cat)  # (N, edge_hidden, D, H, W)
        
        # 边缘预测
        edge_logits = self.edge_head(e_fused)
        
        # 生成供 EAG 使用的多尺度边缘特征（下采样到各级解码器分辨率）
        edge_features = {
            's0': self.edge_proj_s0(e_fused),  # 原始分辨率
            's1': self.edge_proj_s1(F.interpolate(e_fused, scale_factor=0.5, mode='trilinear', align_corners=False)),
            's2': self.edge_proj_s2(F.interpolate(e_fused, scale_factor=0.25, mode='trilinear', align_corners=False)),
            's3': self.edge_proj_s3(F.interpolate(e_fused, scale_factor=0.125, mode='trilinear', align_corners=False)),
        }
        
        return edge_logits, edge_features


class EdgeGuidedSwinUNETR(nn.Module):
    """
    双分支边缘引导 Swin-UNETR (Edge-Guided Swin-UNETR)
    ============================================================
    通过包装 MONAI 的 SwinUNETR，在不修改其内部实现的前提下，
    通过 forward hook 截获编码器的中间特征，
    并在解码器的跳跃连接处注入边缘引导注意力门 (EAG)。
    
    架构示意：
    
    输入 CT → [Swin Transformer Encoder] → 多尺度特征
                    ↓                           ↓
            [Edge Branch] ←──────────── 浅层/中层特征
                    ↓
            edge_logits (边缘预测)
            edge_features (多尺度边缘特征)
                    ↓
            [EAG] ← 注入到每一级跳跃连接
                    ↓
            [CNN Decoder] → seg_logits (分割预测)
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
        spatial_dims=3,
    ):
        super().__init__()
        
        # 1. 实例化原版 SwinUNETR（完整保留，作为骨干）
        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )
        
        self.feature_size = feature_size
        
        # SwinUNETR 编码器各层级的通道数
        # 对于 feature_size=48:
        # encoder0: 48 (原始分辨率 patch embedding)
        # encoder1: 48 (1/2 下采样后的 Swin stage 1)
        # encoder2: 96 (1/4)
        # encoder3: 192 (1/8)
        # encoder4: 384 (1/16)
        # encoder10 (bottleneck): 768 (1/32)
        enc_channels = [
            feature_size,       # 48, s0
            feature_size,       # 48, s1 (1/2)
            feature_size * 2,   # 96, s2 (1/4)
            feature_size * 4,   # 192, s3 (1/8)
            feature_size * 8,   # 384, s4 (1/16)
            feature_size * 16,  # 768, bottleneck (1/32)
        ]
        
        # 2. 边缘检测分支（利用浅层 + 中层编码器特征）
        self.edge_branch = EdgeBranch(enc_channels, out_channels=out_channels)
        
        # 3. 边缘引导注意力门 (EAG)
        # SwinUNETR 解码器内部有 4 级跳跃连接（decoder5 -> decoder4 -> ... -> decoder1）
        # 我们在每一级都注入 EAG
        edge_hidden = 32  # 与 EdgeBranch 的输出维度一致
        
        # decoder5: 接收 encoder4 的 skip (384ch) + bottleneck 上采样 (384ch)
        self.eag_s3 = EdgeAttentionGate(
            F_skip=enc_channels[3],    # 192
            F_up=enc_channels[4],      # 384 (decoder5 输出后通过 decoder4 使用)
            F_edge=edge_hidden,
            F_int=enc_channels[3] // 2  # 96
        )
        
        # decoder4: 接收 encoder3 的 skip (192ch) 
        self.eag_s2 = EdgeAttentionGate(
            F_skip=enc_channels[2],    # 96
            F_up=enc_channels[3],      # 192
            F_edge=edge_hidden,
            F_int=enc_channels[2] // 2  # 48
        )
        
        # decoder3: 接收 encoder1 的 skip (48ch)
        self.eag_s1 = EdgeAttentionGate(
            F_skip=enc_channels[1],    # 48
            F_up=enc_channels[2],      # 96
            F_edge=edge_hidden,
            F_int=enc_channels[1] // 2  # 24
        )
        
        # decoder2: 接收 encoder0 的 skip (48ch)
        self.eag_s0 = EdgeAttentionGate(
            F_skip=enc_channels[0],    # 48
            F_up=enc_channels[1],      # 48
            F_edge=edge_hidden,
            F_int=enc_channels[0] // 2  # 24
        )
        
        # 训练/推理模式标记
        self._training_mode = True
    
    def forward(self, x):
        """
        前向传播：
        1. 通过 Swin Encoder 提取多尺度特征
        2. 将浅层特征送入 Edge Branch 得到边缘预测和边缘特征
        3. 在解码器的每一级跳跃连接处，用 EAG 进行边缘引导的特征增强
        4. 最终输出分割和边缘预测
        """
        target_size = x.shape[2:]  # (D, H, W)
        
        # ============== Step 1: Swin Encoder ==============
        # 使用 SwinUNETR 内部的编码器提取各层级特征
        swin = self.swin_unetr
        
        # Patch Embedding + Position Encoding
        hidden_states = swin.swinViT(x, swin.normalize)
        # hidden_states 是一个列表:
        # [0]: stage0 output (1/2 res, 48ch)
        # [1]: stage1 output (1/4 res, 96ch)  
        # [2]: stage2 output (1/8 res, 192ch)
        # [3]: stage3 output (1/16 res, 384ch)
        # [4]: bottleneck output (1/32 res, 768ch)
        
        # encoder0: 初始 patch embedding 的映射 (full res -> feature_size channels)
        enc0 = swin.encoder1(x)  # (N, 48, D, H, W) 原始分辨率
        
        # encoder stages
        enc1 = swin.encoder2(hidden_states[0])  # (N, 48, D/2, H/2, W/2)
        enc2 = swin.encoder3(hidden_states[1])  # (N, 96, D/4, H/4, W/4)
        enc3 = swin.encoder4(hidden_states[2])  # (N, 192, D/8, H/8, W/8)
        
        # decoder bottleneck
        enc4 = swin.encoder10(hidden_states[4])  # (N, 768, D/32, H/32, W/32)
        
        # ============== Step 2: Edge Branch ==============
        enc_feats = [enc0, enc1, enc2, enc3]
        edge_logits, edge_features = self.edge_branch(enc_feats, target_size)
        
        # ============== Step 3: Decoder with EAG ==============
        # decoder5: bottleneck -> 1/16
        dec5 = swin.decoder5(enc4, hidden_states[3])  # (N, 384, D/16, H/16, W/16)
        
        # decoder4: 1/16 -> 1/8, skip from enc3 (192ch)
        # 在跳跃连接之前，用 EAG 增强 enc3 的特征
        enc3_enhanced = self.eag_s3(enc3, dec5, edge_features['s3'])
        dec4 = swin.decoder4(dec5, enc3_enhanced)  # (N, 192, D/8, H/8, W/8)
        
        # decoder3: 1/8 -> 1/4, skip from enc2 (96ch)
        enc2_enhanced = self.eag_s2(enc2, dec4, edge_features['s2'])
        dec3 = swin.decoder3(dec4, enc2_enhanced)  # (N, 96, D/4, H/4, W/4)
        
        # decoder2: 1/4 -> 1/2, skip from enc1 (48ch)
        enc1_enhanced = self.eag_s1(enc1, dec3, edge_features['s1'])
        dec2 = swin.decoder2(dec3, enc1_enhanced)  # (N, 48, D/2, H/2, W/2)
        
        # decoder1: 1/2 -> full, skip from enc0 (48ch)
        enc0_enhanced = self.eag_s0(enc0, dec2, edge_features['s0'])
        dec1 = swin.decoder1(dec2, enc0_enhanced)  # (N, 48, D, H, W)
        
        # 最终分割预测
        seg_logits = swin.out(dec1)  # (N, out_channels, D, H, W)
        
        # ============== Step 4: Output ==============
        if self.training:
            return seg_logits, edge_logits
        else:
            return seg_logits
