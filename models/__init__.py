import torch
from .VSNet import VSNet
from .cga_unet import CGAUNet3D
from monai.networks.nets import UNet, AttentionUnet

def build_model(config, device="cpu"):
    name = config.get("model_name", "vsnet").lower()
    
    # 提取 patch_size 用于 VSNet (如 96)
    patch_size = config.get("patch_size", (96, 96, 96))
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)
    img_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size

    if name == "vsnet":
        model = VSNet(
            in_channels=1, 
            out_channels=3, 
            img_size=img_size, 
            training=True
        )
    elif name == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        )
    elif name == "vnet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
            dropout=0.2,
        )
    elif name == "attention_unet":
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
        )
    elif name == "cga_unet":
        model = CGAUNet3D(
            in_channels=1,
            num_classes=3,
            features=(32, 64, 128, 256, 512)
        )
    else:
        raise ValueError(f"Unknown model_name: {name}")
        
    return model.to(device)
