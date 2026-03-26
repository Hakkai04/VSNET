import torch
import torch.nn as nn

class AsymmetricTubularBlock(nn.Module):
    """
    管状非对称残差模块：
    结合 3x3x3 卷积与三个方向的一维卷积 (3x1x1, 1x3x1, 1x1x3)，
    专门用于捕捉肝静脉和门静脉末端细长、带方向性的分支结构。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_main = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv_x = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv_y = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0), bias=False)
        self.conv_z = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=False)
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 1x1 卷积用于匹配残差连接的通道数
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        # 融合主体特征与三个方向的管状特征
        out = self.conv_main(x) + self.conv_x(x) + self.conv_y(x) + self.conv_z(x)
        out = self.bn(out)
        out += res
        return self.relu(out)

class ChannelAttention3d(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class DualAttentionBottleneck(nn.Module):
    """双重注意力瓶颈层：结合通道和空间注意力，捕捉全局依赖并抑制噪声"""
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention3d(in_channels)
        self.sa = SpatialAttention3d()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class LYNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        
        # --- Encoder阶段 ---
        filters = [16, 32, 64, 128, 256]
        self.enc1 = AsymmetricTubularBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = AsymmetricTubularBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = AsymmetricTubularBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool3d(2)
        
        self.enc4 = AsymmetricTubularBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool3d(2)
        
        # --- Bottleneck阶段 ---
        self.bottleneck = nn.Sequential(
            AsymmetricTubularBlock(filters[3], filters[4]),
            DualAttentionBottleneck(filters[4])
        )
        
        # --- Decoder阶段 ---
        self.up4 = nn.ConvTranspose3d(filters[4], filters[3], kernel_size=2, stride=2)
        self.dec4 = AsymmetricTubularBlock(filters[3] * 2, filters[3])
        
        self.up3 = nn.ConvTranspose3d(filters[3], filters[2], kernel_size=2, stride=2)
        self.dec3 = AsymmetricTubularBlock(filters[2] * 2, filters[2])
        
        self.up2 = nn.ConvTranspose3d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec2 = AsymmetricTubularBlock(filters[1] * 2, filters[1])
        
        self.up1 = nn.ConvTranspose3d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec1 = AsymmetricTubularBlock(filters[0] * 2, filters[0])
        
        # --- 多任务输出头 (对标 VSNet) ---
        # 1. 主任务：肝静脉(HV)与门静脉(PV)分割
        self.out_seg = nn.Conv3d(filters[0], num_classes, kernel_size=1)
        # 2. 辅助任务1：中心线回归 (Centerline Regression)
        self.out_centerline = nn.Conv3d(filters[0], 1, kernel_size=1)
        # 3. 辅助任务2：边缘分割 (Edge Segmentation)
        self.out_edge = nn.Conv3d(filters[0], 1, kernel_size=1)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # 瓶颈
        b = self.bottleneck(self.pool4(e4))
        
        # 解码与跳跃连接
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # 多任务预测
        seg_out = self.out_seg(d1)
        center_out = self.out_centerline(d1)
        edge_out = self.out_edge(d1)
        
        return seg_out, center_out, edge_out

# 快速测试
if __name__ == "__main__":
    model = LYNet(in_channels=1, num_classes=3)
    # 模拟输入一个 96x96x96 的 3D CT 图像块
    dummy_input = torch.randn(2, 1, 96, 96, 96) 
    seg, center, edge = model(dummy_input)
    
    print("模型输出尺寸检查:")
    print("分割主任务 (Main Seg):", seg.shape)          # 预期: [2, 3, 96, 96, 96]
    print("中心线回归 (Centerline):", center.shape)  # 预期: [2, 1, 96, 96, 96]
    print("边缘分割 (Edge):", edge.shape)            # 预期: [2, 1, 96, 96, 96]