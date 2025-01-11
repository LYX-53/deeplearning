import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
#本代码仅读取数据部分参考已有代码，如已有方法一（基于LSTM+注意力机制(self-attention)进行天气变化的时间序列预测）Time-weather_change_map.py；
# 其他部分均为自己编写,本研究设计了一个新的混合神经网络模型enhanced_hybrid_model。
# 参数设置
timestep = 1
batch_size = 16
input_dim = 14
hidden_dim = 64
output_dim = 1
num_layers = 2
epochs = 10
model_name = 'enhanced_hybrid_model'
save_path = f'./{model_name}.pth'

# 加载和预处理数据
df = pd.read_csv('weather2.csv', index_col=0)
scaler_model = StandardScaler()
data = scaler_model.fit_transform(df.values)

# 数据分割函数
def split_data(data, timestep):
    X, Y = [], []
    for i in range(len(data) - timestep):
        X.append(data[i:i + timestep])
        Y.append(data[i + timestep, 1])
    return np.array(X), np.array(Y)

# 获取训练和测试数据
X, Y = split_data(data, timestep)
train_size = int(0.8 * len(X))
x_train, y_train = X[:train_size], Y[:train_size]
x_test, y_test = X[train_size:], Y[train_size:]

# 转换为张量
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 数据集和数据加载器
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 定义模型
class EnhancedHybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(EnhancedHybridModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden_dim, num_layers, batch_first=True)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # CNN layer
        x = x.permute(0, 2, 1)
        x = self.relu(self.cnn(x))
        x = x.permute(0, 2, 1)

        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # GRU layer
        gru_out, _ = self.gru(lstm_out)

        # Attention layer
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)

        # Fully connected layer
        out = self.fc(self.dropout(attn_out[:, -1, :]))
        return out

# 实例化和训练模型
model = EnhancedHybridModel(input_dim, hidden_dim, num_layers, output_dim)
loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc='Epoch {}:'.format(epoch + 1)):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f'Training Loss: {running_loss / len(train_loader)}')

# 可视化结果
model.eval()
with torch.no_grad():
    predictions = []
    for inputs in test_loader:
        outputs = model(inputs[0])
        predictions.extend(outputs.numpy())
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Temperature Prediction')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()
