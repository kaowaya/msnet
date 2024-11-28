import torch
import torch.nn as nn

class CustomMAEAccuracyLoss(nn.Module):
    def __init__(self, total_capacity):
        super(CustomMAEAccuracyLoss, self).__init__()
        self.total_capacity = total_capacity

    def forward(self, y_true, y_pred):
        # 计算未归一化的 MAE
        mae = torch.mean(torch.abs(y_true - y_pred))
        
        # 计算损失，损失 = MAE / 总装机容量
        loss = mae / self.total_capacity
        
        # 返回损失
        return loss
