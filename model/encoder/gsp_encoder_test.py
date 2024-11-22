import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_results(y_hat, y):

    plt.figure(figsize=(10, 6))
    plt.plot(y_hat[0, 0, :].detach().cpu().numpy(), label="Predicted")
    plt.plot(y[0, 0, :].detach().cpu().numpy(), label="Actual")
    plt.title("Prediction vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


# 定义数据集
class PVDataset(Dataset):
    def __init__(self, data,                     
                gfs_frequency_per_hour,
                gfs_input_time_len_in_hour,
                output_time_len_in_hour,
                predict_interval_in_hour):
        
        self.data = data
        self.gfs_frequency_per_hour = gfs_frequency_per_hour
        self.gfs_input_time_len_in_hour = gfs_input_time_len_in_hour
        self.output_time_len_in_hour = output_time_len_in_hour
        self.predict_interval_in_hour = predict_interval_in_hour

        self.gfs_seq_length = gfs_input_time_len_in_hour*gfs_frequency_per_hour
        self.gfs_forecast_horizon = (output_time_len_in_hour+1)*gfs_frequency_per_hour
        self.gfs_predict_interval = predict_interval_in_hour*gfs_frequency_per_hour

    def __len__(self):
        return (len(self.data) - self.gfs_seq_length - self.gfs_forecast_horizon)//self.gfs_predict_interval + 1

    def __getitem__(self, idx):
        gfs_seq = df[idx*self.gfs_predict_interval : 
                     idx*self.gfs_predict_interval+self.gfs_seq_length].values.astype('float32')
        gfs_label = df[idx*self.gfs_predict_interval+self.gfs_seq_length : 
                       idx*self.gfs_predict_interval+self.gfs_seq_length+self.gfs_forecast_horizon : 
                       4].values.astype('float32')
        return gfs_seq, gfs_label



# 定义模型
class PVModel(pl.LightningModule):
    def __init__(self):
        super(PVModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(96, 72)

        encoder_layers = nn.TransformerEncoderLayer(d_model=72, 
                                                     nhead=36, 
                                                     dim_feedforward=10, 
                                                     dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)


        self.fc2 = nn.Linear(72, (4+1))  # 预测未来4个小时的数据

        self.train_losses = []

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = torch.mean(x, dim=1)  # 全局平均池化
        x = F.relu(self.fc1(x))

        x = self.transformer_encoder(x) #输入和输出的维度都是tf_input_dim
        
        x = self.fc2(x)
        
        x = x.view(-1, 1, 5)
        x = F.relu(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.permute(0, 2, 1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.permute(0, 2, 1)
        y_hat = self(x)

        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.permute(0, 2, 1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        # 可视化预测结果
        plot_results(y_hat, y)

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



# 准备数据
gfs_frequency_per_hour = 4 # 数据每15分钟一帧
gfs_input_time_len_in_hour = 24 # 输入的时长
output_time_len_in_hour = 4 #要预测的时长，比如预测4小时，那就给出5个结果，【0，1，2，3，4】小时的结果
predict_interval_in_hour = 4  # 每四个小时预测一次，每次预测都用预测启动时间点的前24小时作为输入，预测启动时间点的后4小时作为输出
# data = {
#     'generation(kWh)': [i*0.25 for i in range(1000)],  # 用你的实际数据替换...
#     'power(W)': [i*0.25*10 for i in range(1000)]  # 用你的实际数据替换...
# }
# df = pd.DataFrame(data)
df = pd.read_csv('.//exemple_data//gfs//SQ2.csv').head(20000)
df = df[['generation(kWh)']]
# 替换非数字值为 NaN, 前向填充
df = df.applymap(lambda x: np.nan if not isinstance(x, (int, float)) and pd.isna(x) == False and x != '' else x)
df = df.ffill()

dataset = PVDataset(df, 
                    gfs_frequency_per_hour = gfs_frequency_per_hour,
                    gfs_input_time_len_in_hour = gfs_input_time_len_in_hour,
                    output_time_len_in_hour = output_time_len_in_hour,
                    predict_interval_in_hour = predict_interval_in_hour,)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# # 使用for循环迭代数据加载器
# for i,batch in enumerate(dataloader):
#     if i==1:
#         print(batch)  # batch是一个元组，包含(seq, pic)和label
#         break
# # 实例化模型并训练
model = PVModel()
trainer = pl.Trainer(max_epochs=600)
trainer.fit(model, dataloader)

# 测试模型
# test_results = trainer.test(model, dataloader)

# 可视化预测结果
# plot_results(y_hat, y)
 
