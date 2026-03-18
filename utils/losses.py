import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

class VSNetLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.criterion_seg = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False, batch=True)
        self.criterion_edge = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False, batch=True)
        self.criterion_reg = nn.MSELoss()

    def forward(self, outputs, targets):
        seg_v, reg, seg_e, deep2, deep3 = outputs
        labels = targets["label"]
        edges = targets["edge"]
        regs = targets["reg"]

        # 主干 Loss
        L_se = self.criterion_seg(seg_v, labels)
        L_cr = self.criterion_reg(reg, regs)
        L_ec = self.criterion_edge(seg_e, edges)
        
        # 深度监督
        labels_d2 = F.interpolate(labels.float(), size=deep2.shape[2:], mode='nearest').to(labels.dtype)
        labels_d3 = F.interpolate(labels.float(), size=deep3.shape[2:], mode='nearest').to(labels.dtype)
        L_deep = (self.criterion_seg(deep2, labels_d2) + self.criterion_seg(deep3, labels_d3)) / 2.0
        
        # 加权融合
        loss = L_se + self.alpha * L_cr + self.beta * L_ec + self.gamma * L_deep
        return loss, {"seg": L_se.item(), "reg": L_cr.item(), "edge": L_ec.item()}

class StandardLossWrapper(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
        
    def forward(self, outputs, targets):
        labels = targets["label"]
        loss = self.criterion(outputs, labels)
        return loss, {"seg": loss.item()}

def build_loss(config):
    model_name = config.get("model_name", "vsnet").lower()
    
    if model_name == "vsnet":
        return VSNetLoss(
            alpha=config.get("alpha", 1.0),
            beta=config.get("beta", 1.0),
            gamma=config.get("gamma", 0.1)
        )
    else:
        # 其他 Baseline (如 UNet 等) 的普通 Loss
        # 因为我们的模型返回的是单张量的 logits，可以直接套用 DiceCELoss
        criterion = DiceCELoss(
            softmax=True, 
            to_onehot_y=True, 
            include_background=False, 
            batch=True
        )
        return StandardLossWrapper(criterion)
