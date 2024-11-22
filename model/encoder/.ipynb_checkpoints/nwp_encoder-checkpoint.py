
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class Conv3d_NWP(nn.Module):
    def __init__(self, in_channels, 
                 kernel_sizes, 
                 paddings, 
                 channel_num_list,
                 tf_input_dim,
                 nhead, 
                 nhid, 
                 nlayers, 
                 dropout):
        super(Conv3d_NWP, self).__init__()
        # 初始化一个空列表来存储Conv1d层
        layers = []

        # 使用for循环动态添加Conv1d层
        for i in range(len(channel_num_list)):
            layers.append(nn.Conv3d(in_channels=in_channels if i == 0 else channel_num_list[i-1],
                                    out_channels=channel_num_list[i],
                                    kernel_size=(kernel_sizes[i],kernel_sizes[i],kernel_sizes[i]),
                                    padding=(paddings[i],paddings[i],paddings[i])))
            layers.append(nn.ReLU())
        
        # 将列表转换为nn.Sequential
        self.convs = nn.Sequential(*layers)

        self.fc1 = nn.Linear(channel_num_list[-1]*16, tf_input_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=tf_input_dim, 
                                                     nhead=nhead, 
                                                     dim_feedforward=nhid, 
                                                     dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.fc2 = nn.Linear(tf_input_dim, (4+1))  # 预测未来4个小时的数据
        self.train_losses = []



    def forward(self, x): #[b, 10, 5, 4, 6, 6]
        # x = x[1][0] #这行以后去掉
        # print(x.shape)
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4), x.size(5))
        # print('x after view',x.shape)
        x = self.convs(x)
        # print('x after convs',x.shape)
        x = torch.mean(x, dim=2)  # 全局平均池化
        # print('x after 平均池化1',x.shape)
        x = x.view(x.size(0), x.size(1) * x.size(2)* x.size(3))
        #x = torch.mean(x, dim=2)  # 全局平均池化
        # print('x after 平均池化2',x.shape)
        x = F.relu(self.fc1(x))
        x = self.transformer_encoder(x) #输入和输出的维度都是tf_input_dim
        x = self.fc2(x)
        x = x.view(-1, 1, 5)
        x = F.relu(x)
        return x
    
# # 定义卷积核大小和填充大小
# kernel_sizes = [3 for _ in range(3)]  # 示例中添加了3个Conv3d层，每个层的核大小为3
# paddings = [1 for _ in range(3)]  # 填充大小
# out_channels = [4, 8, 64]
# tf_input_dim = 128
# nhead = 32
# nhid = 20
# nlayers = 6
# output_features = 64
# # 创建模型实例
# gfs_time_resolution = 1 # unit: hr
# gfs_time_len = 24
# gfs_predict_len = 4
# gfs_predict_resolution = 1
# gfs_matrix = np.random.rand(2, gfs_time_len)
# gfs_matrix = torch.from_numpy(gfs_matrix).float()
# gfs_matrix = torch.unsqueeze(gfs_matrix, 0) #增加batch维度
# print('input shape',gfs_matrix.shape) #torch.Size([1, 2, 24])

if __name__=='__main__':
    import pandas as pd
    df = pd.read_csv('.//exemple_data//gfs//SQ2.csv').head(3000)
    df = df[['generation(kWh)']]
    # 替换非数字值为 NaN, 前向填充
    df = df.applymap(lambda x: np.nan if not isinstance(x, (int, float)) and pd.isna(x) == False and x != '' else x)
    df = df.ffill()
    gsp_data = np.array(df)

    nwp_data = np.load('.//exemple_data//nwp//nwp_data_202207.npy', allow_pickle=True)

    data = gsp_data,[nwp_data]
    dataset = PV_multi_Dataset(config, data)  
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)    
    # # 实例化模型并训练
    model = NWPModel(in_channels = 50, 
                    kernel_sizes=[3,3,3,3,3,3], 
                    paddings=[0,1,1,1,1,1], 
                    channel_num_list=[150,150,150,150,150,150],
                    tf_input_dim=100,
                    nhead=50, 
                    nhid=10, 
                    nlayers=6, 
                    dropout=0.1)
    trainer = pl.Trainer(max_epochs=200)
    trainer.fit(model, dataloader)

