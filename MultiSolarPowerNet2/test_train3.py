from torch.utils.data import Dataset, DataLoader, Subset
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import torch
 
from util.test_config import Config
from model.multimodel import MultiModel
from util.dataset.multi_dataset import PV_multi_Dataset
from util.test_config import Config


config = Config()
print(config.include_nwp)

gsp_data = np.load('./jq_data/gfs_data.npy')
nwp_data = np.load('./jq_data/nwp_data.npy')


device = 0

fix_seed = 30912
# fix_seed = 71005
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

data = gsp_data,[nwp_data]
dataset = PV_multi_Dataset(config, data)

row1 = int(0.7 * len(dataset))
row2 = int(0.85 * len(dataset))

train_dataset = Subset(dataset,list(range(row1)))
print(f'训练集长度为{len(train_dataset)}')
vali_dataset = Subset(dataset,list(range(row1,row2)))
print(f'验证集长度为{len(vali_dataset)}')
test_dataset = Subset(dataset,list(range(row2,len(dataset))))
print(f'测试集长度为{len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
vali_loader = DataLoader(vali_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # 监控验证损失
    dirpath='checkpoints',  # 保存模型的目录
    filename='best_model',  # 文件名模板
    save_top_k=1,  # 只保存最好的模型
    mode='min',  # 选择最小的验证损失
    verbose=True  # 启用日志
)

class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_loss.append(trainer.callback_metrics["val_loss"].item())

loss_history = LossHistory()

model = MultiModel(config)#.to(device)
trainer = pl.Trainer(max_epochs=500,callbacks=[loss_history,checkpoint_callback])

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=vali_loader)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(loss_history.train_loss, label='Train Loss')
plt.plot(loss_history.val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss曲线')
plt.show()

best_model_path = trainer.checkpoint_callback.best_model_path
test_model = MultiModel.load_from_checkpoint(checkpoint_path=best_model_path, config=config)
test_model.eval()

def model_predict(model, loader):
    model.cpu()  # 确保模型在正确的设备上
    predictions = []
    truth = []
    for i,t in loader:
        model.eval()
        y_pre = model(i)
        predictions.append(y_pre.cpu().detach().numpy())
        truth.append(t.cpu().detach().numpy())
    
    predictions = np.concatenate(predictions)
    truth = np.concatenate(truth)
    return predictions, truth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_model.to(device)
train_predictions,train_truth = model_predict(test_model,train_loader)
test_predictions,test_truth = model_predict(test_model,test_loader)

train_predictions = np.squeeze(train_predictions)
train_truth = np.squeeze(train_truth)
test_predictions = np.squeeze(test_predictions)
test_truth = np.squeeze(test_truth)

t1 = np.delete(train_truth, 0, axis=1).reshape(-1,1)
y1 = np.delete(train_predictions, 0, axis=1).reshape(-1,1)
t2 = np.delete(test_truth, 0, axis=1).reshape(-1,1)
y2 = np.delete(test_predictions, 0, axis=1).reshape(-1,1)

from scripts.getpic import train_test_pic, compare_pic

print('测试集与训练集比较：')
train_test_pic(y1,t1,y2,t2)

print('测试集与现有模型比较：')
compare_pic(y2,t2)
