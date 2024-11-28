


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class BaseModel(pl.LightningModule):
    def __init__(self ):
        super(BaseModel, self).__init__()
        self.train_losses = []
        self.val_losses = []
        # 使用自定义损失函数
        # self.loss_fn = CustomMAEAccuracyLoss(total_capacity=self.total_capacity)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.permute(0, 2, 1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        # loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.permute(0, 2, 1)
        y_hat = self(x)

        loss = F.mse_loss(y_hat, y)
        # loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.permute(0, 2, 1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        # loss = self.loss_fn(y_hat, y)
        # 可视化预测结果
        self.plot_results(y_hat, y)

        self.log('test_loss', loss)
        return {"y_hat": y_hat, "y": y}

    def test_end(self, outputs):
        # outputs 是一个列表，包含了 test_step 方法返回的所有损失值
        # 我们在这里处理 y_hat 和 y
        y_hats, ys = [], []
        for output in outputs:
            # 假设我们在 test_step 方法中将 y_hat 和 y 存储在属性中
            # 注意：这需要你在 test_step 方法中保存这些值
            y_hats.append(self.y_hat.detach())
            ys.append(self.y.detach())

        # 将列表转换为张量
        y_hats = torch.cat(y_hats)
        ys = torch.cat(ys)
        return {"y_hats": y_hats, "ys": ys}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def plot_results(self, y_hat, y):
        import matplotlib as plt
        plt.figure(figsize=(10, 6))
        plt.plot(y_hat[0, 0, :].detach().cpu().numpy(), label="Predicted")
        plt.plot(y[0, 0, :].detach().cpu().numpy(), label="Actual")
        plt.title("Prediction vs Actual")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.mse_loss(y_hat, y)
    #     self.log('train_loss', loss)
    #     self.train_losses.append(loss.item())
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.mse_loss(y_hat, y)
    #     self.log('val_loss', loss)
    #     self.val_losses.append(loss.item())

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.mse_loss(y_hat, y)
    #     self.log('test_loss', loss)
 

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    #     return optimizer
    
    def _quantiles_to_prediction(self, y_quantiles):
        """
        Convert network prediction into a point prediction.

        Note:
            Implementation copied from:
                https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting
                /metrics/quantile.html#QuantileLoss.loss

        Args:
            y_quantiles: Quantile prediction of network

        Returns:
            torch.Tensor: Point prediction
        """
        # y_quantiles Shape: batch_size, seq_length, num_quantiles
        idx = self.output_quantiles.index(0.5)
        y_median = y_quantiles[..., idx]
        return y_median
    
    def _calculate_quantile_loss(self, y_quantiles, y):
        """Calculate quantile loss.

        Note:
            Implementation copied from:
                https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting
                /metrics/quantile.html#QuantileLoss.loss

        Args:
            y_quantiles: Quantile prediction of network
            y: Target values

        Returns:
            Quantile loss
        """
        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.output_quantiles):
            errors = y - y_quantiles[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)
        if self.use_weighted_loss:
            weights = self.weighted_losses.weights.unsqueeze(1).unsqueeze(0).to(y.device)
            losses = losses * weights
        return losses.mean()

    def _calculate_common_losses(self, y, y_hat):
        """Calculate losses common to train, test, and val"""

        losses = {}

        if self.use_quantile_regression:
            losses["quantile_loss"] = self._calculate_quantile_loss(y_hat, y)
            y_hat = self._quantiles_to_prediction(y_hat)

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)

        # calculate mse, mae with exp weighted loss
        mse_exp = self.weighted_losses.get_mse_exp(output=y_hat, target=y)
        mae_exp = self.weighted_losses.get_mae_exp(output=y_hat, target=y)

        # TODO: Compute correlation coef using np.corrcoef(tensor with
        # shape (2, num_timesteps))[0, 1] on each example, and taking
        # the mean across the batch?
        losses.update(
            {
                "MSE": mse_loss,
                "MAE": mae_loss,
                "MSE_EXP": mse_exp,
                "MAE_EXP": mae_exp,
            }
        )

        return losses

    def _step_mae_and_mse(self, y, y_hat, dict_key_root):
        """Calculate the MSE and MAE at each forecast step"""
        losses = {}

        mse_each_step = torch.mean((y_hat - y) ** 2, dim=0)
        mae_each_step = torch.mean(torch.abs(y_hat - y), dim=0)

        losses.update({f"MSE_{dict_key_root}/step_{i:03}": m for i, m in enumerate(mse_each_step)})
        losses.update({f"MAE_{dict_key_root}/step_{i:03}": m for i, m in enumerate(mae_each_step)})

        return losses

    def _calculate_val_losses(self, y, y_hat):
        """Calculate additional validation losses"""

        losses = {}

        if self.use_quantile_regression:
            # Add fraction below each quantile for calibration
            for i, quantile in enumerate(self.output_quantiles):
                below_quant = y <= y_hat[..., i]
                # Mask values small values, which are dominated by night
                mask = y >= 0.01
                losses[f"fraction_below_{quantile}_quantile"] = (below_quant[mask]).float().mean()

            # Take median value for remaining metric calculations
            y_hat = self._quantiles_to_prediction(y_hat)

        # Log the loss at each time horizon
        losses.update(self._step_mae_and_mse(y, y_hat, dict_key_root="horizon"))

        # Log the persistance losses
        y_persist = y[:, -1].unsqueeze(1).expand(-1, self.forecast_len)
        losses["MAE_persistence/val"] = F.l1_loss(y_persist, y)
        losses["MSE_persistence/val"] = F.mse_loss(y_persist, y)

        # Log persistance loss at each time horizon
        losses.update(self._step_mae_and_mse(y, y_persist, dict_key_root="persistence"))
        return losses

    def _calculate_test_losses(self, y, y_hat):
        """Calculate additional test losses"""
        # No additional test losses
        losses = {}
        return losses

    def _training_accumulate_log(self, batch, batch_idx, losses, y_hat):
        """Internal function to accumulate training batches and log results.

        This is used when accummulating grad batches. Should make the variability in logged training
        step metrics indpendent on whether we accumulate N batches of size B or just use a larger
        batch size of N*B with no accumulaion.
        """

        losses = {k: v.detach().cpu() for k, v in losses.items()}
        y_hat = y_hat.detach().cpu()

        self._accumulated_metrics.append(losses)
        self._accumulated_batches.append(batch)
        self._accumulated_y_hat.append(y_hat)

        if not self.trainer.fit_loop._should_accumulate():
            losses = self._accumulated_metrics.flush()
            batch = self._accumulated_batches.flush()
            y_hat = self._accumulated_y_hat.flush()

            self.log_dict(
                losses,
                on_step=True,
                on_epoch=True,
            )

            # Number of accumulated grad batches
            grad_batch_num = (batch_idx + 1) / self.trainer.accumulate_grad_batches

            # We only create the figure every 8 log steps
            # This was reduced as it was creating figures too often
            if grad_batch_num % (8 * self.trainer.log_every_n_steps) == 0:
                pass
                # fig = plot_batch_forecasts(
                #     batch,
                #     y_hat,
                #     batch_idx,
                #     quantiles=self.output_quantiles,
                #     key_to_plot=self._target_key_name,
                # )
                # fig.savefig("latest_logged_train_batch.png")
                # plt.close(fig)
