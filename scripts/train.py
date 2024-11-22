import torch
import torch.nn as nn

# todo: dataloader


# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 实例化模型
model = SCC(input_channels=1, hidden_channels=64, output_features=1)

# 训练模型
trainer = pl.Trainer(max_epochs=10, callbacks=[ModelCheckpoint(monitor='val_loss', save_top_k=1)])
trainer.fit(model, train_loader, val_loader)

# 测试模型
# 假设test_dataset是已经定义好的测试数据集
test_dataset = TimeSeriesDataset(data[:10], labels[:10])  # 示例：使用部分数据作为测试集
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
trainer.test(model, test_loader)