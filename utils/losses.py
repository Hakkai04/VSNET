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

class SoftClDiceLoss3D(nn.Module):
    def __init__(self, iter_=3, smooth=1e-5):
        super(SoftClDiceLoss3D, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def soft_erode(self, img):
        p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0))
        p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        return F.max_pool3d(img, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for _ in range(self.iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, y_true, y_pred):
        cldice = 0
        for c in range(1, y_pred.shape[1]):
            skel_pred = self.soft_skel(y_pred[:, c:c+1])
            skel_true = self.soft_skel(y_true[:, c:c+1])
            
            tprec = (torch.sum(skel_pred * y_true[:, c:c+1]) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
            tsens = (torch.sum(skel_true * y_pred[:, c:c+1]) + self.smooth) / (torch.sum(skel_true) + self.smooth)
            
            cl_dice_c = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
            cldice += cl_dice_c
            
        return cldice / (y_pred.shape[1] - 1)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, iter_=3, warmup_epochs=500):
        super().__init__()
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs
        self.dice_ce = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False, batch=True)
        self.cldice = SoftClDiceLoss3D(iter_=iter_)

    def forward(self, outputs, targets):
        labels = targets["label"]
        epoch = targets.get("epoch", 1)  # 获取当前 epoch

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        loss_dice_ce = self.dice_ce(outputs, labels)

        # 缓冲机制：前 warmup_epochs 轮只使用 dice_ce，不计算 cldice
        if epoch <= self.warmup_epochs:
            loss = loss_dice_ce
            return loss, {"dice_ce": loss_dice_ce.item(), "cldice": 0.0, "total": loss.item()}

        probs = F.softmax(outputs, dim=1)
        labels_onehot = F.one_hot(labels.squeeze(1).long(), num_classes=outputs.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        loss_cldice = self.cldice(labels_onehot, probs)

        loss = self.alpha * loss_dice_ce + (1.0 - self.alpha) * loss_cldice

        return loss, {"dice_ce": loss_dice_ce.item(), "cldice": loss_cldice.item(), "total": loss.item()}

class StandardLossWrapper(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
        
    def forward(self, outputs, targets):
        labels = targets["label"]
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
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
    elif model_name == "attention_unet":
        return CombinedLoss(
            alpha=config.get("alpha", 0.5),
            iter_=3
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
