
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class Conv1d_GSP(nn.Module):
    def __init__(self, in_channels, 
                 kernel_sizes, 
                 paddings, 
                 channel_num_list,
                 tf_input_dim,
                 nhead, 
                 nhid, 
                 nlayers, 
                 dropout):
        super(Conv1d_GSP, self).__init__()
        
        # 初始化一个空列表来存储Conv1d层
        layers = []

        # 使用for循环动态添加Conv1d层
        for i in range(len(channel_num_list)):
            layers.append(nn.Conv1d(in_channels=in_channels if i == 0 else channel_num_list[i-1],
                                    out_channels=channel_num_list[i],
                                    kernel_size=(kernel_sizes[i]),
                                    padding=(paddings[i])))
            layers.append(nn.ReLU())
        
        # 将列表转换为nn.Sequential
        self.convs = nn.Sequential(*layers)

        self.fc1 = nn.Linear(96, tf_input_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=tf_input_dim, 
                                                     nhead=nhead, 
                                                     dim_feedforward=nhid, 
                                                     dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.fc2 = nn.Linear(tf_input_dim, (4+1))  # 预测未来4个小时的数据



    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.convs(x)
        x = torch.mean(x, dim=1)  # 全局平均池化
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
    batch_num = 16
    input_size = 96
    in_channels = 1
    channel_num_list = [4,8,16,32,32,32]
    kernel_sizes = [3]*len(channel_num_list)
    paddings = [1]*len(channel_num_list)
    tf_input_dim = 72
    nhead = 36
    nhid = 10
    nlayers=6
    conv1d_gsp = Conv1d_GSP(in_channels=in_channels,  
                        kernel_sizes=kernel_sizes,
                        paddings=paddings, 
                        channel_num_list =channel_num_list, 
                        tf_input_dim = tf_input_dim,
                        nhead=nhead, 
                        nhid=nhid, 
                        nlayers=nlayers, 
                        dropout=0.3)
    
    gsp_matrix = np.random.rand(batch_num, input_size, in_channels)  
    gsp_matrix = torch.from_numpy(gsp_matrix).float()
    output = conv1d_gsp(gsp_matrix)
    print('output shape',output.shape)

    

