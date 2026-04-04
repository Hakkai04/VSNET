import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete

class Trainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scaler, 
        config, 
        logger, 
        writer, 
        save_dir, 
        device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.logger = logger
        self.writer = writer
        self.save_dir = save_dir
        self.weights_dir = os.path.join(save_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        self.device = device
        
        self.max_epochs = config.get("max_epochs", 2000)
        self.val_interval = config.get("val_interval", 5)
        
        self.batch_size = config.get("batch_size", 4)
        self.target_batch_size = config.get("target_batch_size", 16)
        self.accumulation_steps = self.target_batch_size // self.batch_size
        if self.target_batch_size % self.batch_size != 0:
            self.logger.warning(f"⚠️ Target batch size {self.target_batch_size} implies fractional accumulation with batch size {self.batch_size}.")
        
        patch_size = config.get("patch_size", (96, 96, 96))
        self.patch_size = tuple(patch_size) if isinstance(patch_size, list) else patch_size
        
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
        # self.hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch", get_not_nans=False)
        # self.conf_matrix_metric = ConfusionMatrixMetric(
        #     include_background=False, 
        #     metric_name=["precision", "sensitivity"], 
        #     reduction="mean_batch", 
        #     get_not_nans=False
        # )
        self.best_dice = -1.0
        self.best_epoch = -1
        self.patience = config.get("patience", 2000)

    def train_one_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.max_epochs}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        running_loss = 0.0
        steps = 0
        
        self.optimizer.zero_grad()
        
        for i, batch_data in enumerate(pbar):
            steps += 1
            
            # 准备数据，不同模型需要的数据在 Transforms 阶段已准备好
            inputs = batch_data["image"].to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch_data.items() if k != "image"}

            with torch.amp.autocast('cuda'):
                outputs = self.model(inputs)
                loss, loss_dict = self.criterion(outputs, targets)
                
                # 梯度累积
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.accumulation_steps == 0:
                # 添加梯度裁剪以稳定训练
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            current_loss = loss.item() * self.accumulation_steps
            running_loss += current_loss
            
            # 日志显示
            postfix_dict = {"Loss": f"{current_loss:.4f}"}
            for k, v in loss_dict.items():
                postfix_dict[k] = f"{v:.4f}"
            pbar.set_postfix(postfix_dict)

        if (len(self.train_loader) % self.accumulation_steps) != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        avg_loss = running_loss / steps
        self.writer.add_scalar("Train/Total_Loss", avg_loss, epoch)
        
        # 随时保存最后一个 epoch
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(ckpt, os.path.join(self.weights_dir, "last.pth"))
        return avg_loss

    def validate_one_epoch(self, epoch):
        self.model.eval()
        vis_dir = os.path.join(self.save_dir, "vis_check")
        os.makedirs(vis_dir, exist_ok=True)

        with torch.no_grad():
            val_tqdm = tqdm(self.val_loader, desc="Validating", leave=False)
            for i, val_data in enumerate(val_tqdm):
                val_inputs = val_data["image"].to(self.device, non_blocking=True)
                val_labels = val_data["label"].to(self.device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs, 
                        roi_size=self.patch_size, 
                        sw_batch_size=8, 
                        predictor=self.model,
                        overlap=0.5
                    )
                
                # VSNet 的推理有时返回单个元组，需要解析
                if isinstance(val_outputs, tuple):
                    val_outputs = val_outputs[0]
                    
                # 可视化第一张图
                if i == 0:
                    slice_idx = val_inputs.shape[-1] // 2
                    img_show = val_inputs[0, 0, :, :, slice_idx].cpu().numpy()
                    lbl_show = val_labels[0, 0, :, :, slice_idx].cpu().numpy()
                    pred_show = torch.argmax(val_outputs, dim=1)[0, :, :, slice_idx].detach().cpu().numpy()

                    plt.figure(figsize=(12, 4), dpi=100)
                    plt.subplot(1, 3, 1); plt.imshow(img_show, cmap="gray"); plt.title("Image"); plt.axis('off')
                    plt.subplot(1, 3, 2); plt.imshow(lbl_show, cmap="jet", interpolation='nearest'); plt.title("Label"); plt.axis('off')
                    plt.subplot(1, 3, 3); plt.imshow(pred_show, cmap="jet", interpolation='nearest'); plt.title(f"Pred E{epoch}"); plt.axis('off')
                    plt.savefig(os.path.join(vis_dir, f"epoch_{epoch}_check.png"))
                    plt.close()

                val_outputs_onehot = AsDiscrete(argmax=True, to_onehot=3, dim=1)(val_outputs)
                val_labels_onehot = AsDiscrete(to_onehot=3, dim=1)(val_labels)
                self.dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)
                # self.hd95_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)
                # self.conf_matrix_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)

            dice_score = self.dice_metric.aggregate()
            dice_hv = dice_score[0].item()
            dice_pv = dice_score[1].item()
            mean_dice = torch.nanmean(dice_score).item()
            self.dice_metric.reset()

            # hd95_score = self.hd95_metric.aggregate()
            # hd95_hv = hd95_score[0].item()
            # hd95_pv = hd95_score[1].item()
            # mean_hd95 = torch.nanmean(hd95_score).item()
            # self.hd95_metric.reset()

            # conf_metrics = self.conf_matrix_metric.aggregate()
            # if isinstance(conf_metrics, (tuple, list)):
            #     precision_hv = conf_metrics[0][0].item()
            #     precision_pv = conf_metrics[0][1].item()
            #     sensitivity_hv = conf_metrics[1][0].item()
            #     sensitivity_pv = conf_metrics[1][1].item()
            #     precision_score = torch.nanmean(conf_metrics[0]).item()
            #     sensitivity_score = torch.nanmean(conf_metrics[1]).item()
            # else:
            #     precision_hv = conf_metrics[0].item()
            #     precision_pv = conf_metrics[1].item()
            #     sensitivity_hv = 0.0
            #     sensitivity_pv = 0.0
            #     precision_score = torch.nanmean(conf_metrics).item()
            #     sensitivity_score = 0.0
            # self.conf_matrix_metric.reset()

            self.logger.info(
                f"\nEpoch {epoch} | Val Mean Dice: {mean_dice:.4f} | HV: {dice_hv:.4f} | PV: {dice_pv:.4f}"
                # f"\n        | Val Mean 95HD: {mean_hd95:.4f} | HV: {hd95_hv:.4f} | PV: {hd95_pv:.4f}\n"
                # f"        | Val Mean Prec: {precision_score:.4f} | HV: {precision_hv:.4f} | PV: {precision_pv:.4f}\n"
                # f"        | Val Mean Sens: {sensitivity_score:.4f} | HV: {sensitivity_hv:.4f} | PV: {sensitivity_pv:.4f}"
            )
            self.writer.add_scalar("Val/Mean_Dice", mean_dice, epoch)
            # self.writer.add_scalar("Val/Mean_95HD", mean_hd95, epoch)
            # self.writer.add_scalar("Val/Mean_Precision", precision_score, epoch)
            # self.writer.add_scalar("Val/Mean_Sensitivity", sensitivity_score, epoch)

            if mean_dice > self.best_dice:
                self.best_dice = mean_dice
                self.best_epoch = epoch
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'dice': mean_dice,
                }
                torch.save(ckpt, os.path.join(self.weights_dir, "best.pth"))
                self.logger.info(f"🏆 New Best Model saved! (Dice: {self.best_dice:.4f})")

    def fit(self):
        # 增加学习率衰减 (余弦退火)
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epochs, eta_min=1e-6)

        start_epoch = getattr(self, 'start_epoch', 1)
        
        # 恢复已有的 scheduler 进度
        if start_epoch > 1:
            for _ in range(1, start_epoch):
                scheduler.step()

        for epoch in range(start_epoch, self.max_epochs + 1):
            self.train_one_epoch(epoch)
            scheduler.step()
            
            if epoch % self.val_interval == 0:
                self.validate_one_epoch(epoch)
                
                # Check Early Stopping
                if self.best_epoch > 0 and (epoch - self.best_epoch) >= self.patience:
                    self.logger.info(f"⚠️ Early stopping triggered! No improvement in {self.patience} epochs.")
                    break
                
            # 清理当前 epoch 产生的未引用显存和内存垃圾
            # 避免 fork 子进程时继承悬空 CUDA Tensor 引发 'c10::AcceleratorError' 崩溃
            import gc
            gc.collect()
            torch.cuda.empty_cache()
                
        self.logger.info(f"\n🎉 Training completed. Best Dice: {self.best_dice:.4f} at epoch {self.best_epoch}")
        self.writer.close()
