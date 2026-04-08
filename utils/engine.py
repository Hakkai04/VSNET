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
        self.dice_metric_raw = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
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
        
        from monai.metrics import HausdorffDistanceMetric
        self.hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch", get_not_nans=False)
        from utils.losses import SoftClDiceLoss3D
        self.cldice_calc = SoftClDiceLoss3D() # Loss value is 1 - SoftClDice, so Score = 1 - Loss

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
            targets["epoch"] = epoch  # 传入 epoch 以支持 Loss 缓冲机制

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
                
                # Unwrap if model returns tuple in eval mode (though CGAUNet3D only returns mask in eval)
                if isinstance(val_outputs, tuple):
                    val_outputs = val_outputs[0]
                    
                # We removed KeepLargestConnectedComponent directly using argmax
                val_preds_argmax = torch.argmax(val_outputs, dim=1, keepdim=True)
                val_outputs_post = val_preds_argmax # no component dropping

                # Visualization (first batch only)
                if i == 0:
                    slice_idx = val_inputs.shape[-1] // 2
                    img_show = val_inputs[0, 0, :, :, slice_idx].cpu().numpy()
                    lbl_show = val_labels[0, 0, :, :, slice_idx].cpu().numpy()
                    pred_show = val_outputs_post[0, 0, :, :, slice_idx].detach().cpu().numpy()

                    plt.figure(figsize=(12, 4), dpi=100)
                    plt.subplot(1, 3, 1); plt.imshow(img_show, cmap="gray"); plt.title("Image"); plt.axis('off')
                    plt.subplot(1, 3, 2); plt.imshow(lbl_show, cmap="jet", interpolation='nearest'); plt.title("Label"); plt.axis('off')
                    plt.subplot(1, 3, 3); plt.imshow(pred_show, cmap="jet", interpolation='nearest'); plt.title(f"Pred E{epoch}"); plt.axis('off')
                    plt.savefig(os.path.join(vis_dir, f"epoch_{epoch}_check.png"))
                    plt.close()

                # Metric Computation (One-hot encoding)
                val_outputs_onehot = AsDiscrete(to_onehot=3, dim=1)(val_outputs_post)
                val_labels_onehot = AsDiscrete(to_onehot=3, dim=1)(val_labels)
                
                self.dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)
                
                # Compute clDice using SoftClDiceLoss3D
                probs = F.softmax(val_outputs, dim=1)
                labels_onehot_float = F.one_hot(val_labels.squeeze(1).long(), num_classes=val_outputs.shape[1]).permute(0, 4, 1, 2, 3).float()
                cldice_loss = self.cldice_calc(labels_onehot_float, probs)
                cldice_score = 1.0 - cldice_loss.item()
                if not hasattr(self, 'epoch_cldice_list'):
                    self.epoch_cldice_list = []
                self.epoch_cldice_list.append(cldice_score)

            # DICE Aggregate
            dice_score = self.dice_metric.aggregate()
            dice_hv = dice_score[0].item()
            dice_pv = dice_score[1].item()
            mean_dice = torch.nanmean(dice_score).item()
            self.dice_metric.reset()

            mean_cldice = sum(self.epoch_cldice_list) / max(len(self.epoch_cldice_list), 1)
            self.epoch_cldice_list = []

            self.logger.info(
                f"\nEpoch {epoch} | Mean Dice: {mean_dice:.4f} | HV: {dice_hv:.4f} | PV: {dice_pv:.4f} | clDice: {mean_cldice:.4f}"
            )
            self.writer.add_scalar("Val/Mean_Dice", mean_dice, epoch)
            self.writer.add_scalar("Val/Mean_clDice", mean_cldice, epoch)

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
                
            # 处理内存
            import gc
            gc.collect()
            torch.cuda.empty_cache()
                
        self.logger.info(f"\n🎉 Training completed. Best Dice: {self.best_dice:.4f} at epoch {self.best_epoch}")
        
        # Finally, Evaluate Best Model on HD95 Metric
        self._evaluate_best_model_hd95()
        
        self.writer.close()

    def _evaluate_best_model_hd95(self):
        best_path = os.path.join(self.weights_dir, "best.pth")
        if not os.path.exists(best_path):
            return
            
        self.logger.info(f"🔍 Loading Best Model from {best_path} to calculate HD95...")
        self.model.load_state_dict(torch.load(best_path, map_location=self.device)['model_state_dict'])
        self.model.eval()
        self.hd95_metric.reset()

        with torch.no_grad():
            eval_tqdm = tqdm(self.val_loader, desc="HD95 Eval", leave=False)
            for i, val_data in enumerate(eval_tqdm):
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
                if isinstance(val_outputs, tuple):
                    val_outputs = val_outputs[0]
                    
                val_preds_argmax = torch.argmax(val_outputs, dim=1, keepdim=True)
                val_outputs_onehot = AsDiscrete(to_onehot=3, dim=1)(val_preds_argmax)
                val_labels_onehot = AsDiscrete(to_onehot=3, dim=1)(val_labels)
                
                # compute hd95
                self.hd95_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)

        hd95_score = self.hd95_metric.aggregate()
        hd95_hv = hd95_score[0].item()
        hd95_pv = hd95_score[1].item()
        mean_hd95 = torch.nanmean(hd95_score).item()
        
        self.logger.info(
            f"🎯 Final Best Model Evaluation | Mean HD95: {mean_hd95:.4f} | HV: {hd95_hv:.4f} | PV: {hd95_pv:.4f}"
        )
        self.hd95_metric.reset()
