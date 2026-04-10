import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class CGAGate(nn.Module):
    """
    Centerline-Guided Attention Gate
    Takes features from encoder (x), mask decoder (g_mask), and centerline decoder (g_centerline).
    """
    def __init__(self, in_channels_mask, in_channels_centerline, in_channels_x, inter_channels):
        super().__init__()
        # W_g operates on the concatenated features of mask and transformed centerline
        self.W_g = nn.Conv3d(in_channels_mask + in_channels_centerline, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # W_x operates on the encoder skip connection features
        self.W_x = nn.Conv3d(in_channels_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Learnable Gaussian Decay parameter for Attention Probability transformation
        self.alpha_c = nn.Parameter(torch.tensor(0.5))
        
        # psi maps to attention scores
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g_mask, g_centerline, x):
        if g_mask.shape[2:] != x.shape[2:]:
            g_mask = F.interpolate(g_mask, size=x.shape[2:], mode="trilinear", align_corners=False)
            g_centerline = F.interpolate(g_centerline, size=x.shape[2:], mode="trilinear", align_corners=False)
            
        # [Pitfall Protection] Gaussian Inversion: 
        # Convert absolute distance-like spatial embeddings into properly scaled normalized probability-like Attention Fields
        g_centerline_gaussian = torch.exp(-torch.abs(self.alpha_c) * g_centerline)
            
        # Concat deep features from both decoders
        g = torch.cat([g_mask, g_centerline_gaussian], dim=1)
        
        # Linear transformations
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Fusion
        psi = self.relu(g1 + x1)
        psi_weight = self.psi(psi)
        
        # Attention masking
        return x * psi_weight

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_ch, out_ch) # input will be concatenated with skip

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UpBlockWithCGA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        
        # CGA Gate: inputs are g_mask (in_ch), g_centerline (in_ch), x (out_ch)
        self.cga = CGAGate(in_channels_mask=in_ch, in_channels_centerline=in_ch, in_channels_x=out_ch, inter_channels=out_ch // 2)
        
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, x_mask, x_centerline, skip):
        x_mask_up = self.up(x_mask)
        
        # Attention weighting on skip connection
        skip_attended = self.cga(g_mask=x_mask, g_centerline=x_centerline, x=skip)
        
        x_mask_cat = torch.cat([x_mask_up, skip_attended], dim=1)
        x_mask_out = self.conv(x_mask_cat)
        
        return x_mask_out

class CGAUNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, features=(32, 64, 128, 256, 512)):
        super().__init__()
        
        # Shared Encoder
        self.enc1 = ConvBlock3D(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = ConvBlock3D(features[0], features[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = ConvBlock3D(features[1], features[2])
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc4 = ConvBlock3D(features[2], features[3])
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = ConvBlock3D(features[3], features[4])

        # Centerline Decoder (Standard UpBlocks)
        self.c_up4 = UpBlock3D(features[4], features[3])
        self.c_up3 = UpBlock3D(features[3], features[2])
        self.c_up2 = UpBlock3D(features[2], features[1])
        self.c_up1 = UpBlock3D(features[1], features[0])
        # [Pitfall Protection] Output 1 Channel for absolute distance regression with Sigmoid to bound dynamically mapped [0, 1] T-EDT scale
        self.c_out = nn.Sequential(
            nn.Conv3d(features[0], 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Mask Decoder (UpBlocks with CGA)
        self.m_up4 = UpBlockWithCGA(features[4], features[3])
        self.m_up3 = UpBlockWithCGA(features[3], features[2])
        self.m_up2 = UpBlockWithCGA(features[2], features[1])
        self.m_up1 = UpBlockWithCGA(features[1], features[0])
        self.m_out = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        x = self.pool2(x2)
        
        x3 = self.enc3(x)
        x = self.pool3(x3)
        
        x4 = self.enc4(x)
        x = self.pool4(x4)
        
        bottle = self.bottleneck(x)
        
        # Centerline Decoder Pathway (Independent of Mask)
        c4 = self.c_up4(bottle, x4)
        c3 = self.c_up3(c4, x3)
        c2 = self.c_up2(c3, x2)
        c1 = self.c_up1(c2, x1)
        pred_centerline = self.c_out(c1)
        
        # Mask Decoder Pathway (Guided by Centerline)
        # Bottle level: centerline feature is 'bottle', mask feature is 'bottle'
        m4 = self.m_up4(x_mask=bottle, x_centerline=bottle, skip=x4)
        m3 = self.m_up3(x_mask=m4, x_centerline=c4, skip=x3)
        m2 = self.m_up2(x_mask=m3, x_centerline=c3, skip=x2)
        m1 = self.m_up1(x_mask=m2, x_centerline=c2, skip=x1)
        pred_mask = self.m_out(m1)
        
        if not self.training:
            return pred_mask
            
        return pred_mask, pred_centerline
