import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tushare as ts
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm
#本代码参考网上已有的代码https://weibaohang.blog.csdn.net/article/details/127270853，并结合实际内容修改了输出结果气温预测图部分内容
timestep = 1  # 时间步长，就是利用多少时间窗口
batch_size = 16  # 批次大小
input_dim = 14  # 每个步长对应的特征数量，就是使用每天的4个特征，最高、最低、开盘、落盘
hidden_dim = 64  # 隐层大小
output_dim = 1  # 由于是回归任务，最终输出层大小为1
num_layers = 3  # LSTM的层数
epochs = 10
best_loss = 0
model_name = 'LSTM_ATTENTION'
save_path = './{}.pth'.format(model_name)

# 1.加载时间序列数据
df = pd.read_csv('jena_climate_2009_2016.csv', index_col = 0)

# 2.将数据进行标准化
scaler = StandardScaler()
scaler_model = StandardScaler()
data = scaler_model.fit_transform(np.array(df))
scaler.fit_transform(np.array(df['T (degC)']).reshape(-1, 1))

# 形成训练数据，例如12345变成12-3，23-4，34-5
def split_data(data, timestep, input_dim):
    dataX = []  # 保存X
    dataY = []  # 保存Y

    # 将整个窗口的数据保存到X中，将未来一天保存到Y中
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep])
        dataY.append(data[index + timestep][1])

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # 获取训练集大小
    train_size = int(np.round(0.8 * dataX.shape[0]))

    # 划分训练集、测试集
    x_train = dataX[: train_size, :].reshape(-1, timestep, input_dim)
    y_train = dataY[: train_size]

    x_test = dataX[train_size:, :].reshape(-1, timestep, input_dim)
    y_test = dataY[train_size:]

    return [x_train, y_train, x_test, y_test]

# 3.获取训练数据   x_train: 1700,1,4
x_train, y_train, x_test, y_test = split_data(data, timestep, input_dim)

# 4.将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

# 5.形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size,
                                           True)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size,
                                          False)

# 7.定义LSTM网络
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim  # 隐层大小
        self.num_layers = num_layers  # LSTM层数
        # embed_dim为每个时间步对应的特征数
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=2)
        # input_dim为特征维度，就是每个时间点对应的特征数量，这里为4
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, query, key, value):
#         print(query.shape) # torch.Size([16, 1, 4]) batch_size, time_step, input_dim
        attention_output, attn_output_weights = self.attention(query, key, value)
#         print(attention_output.shape) # torch.Size([16, 1, 4]) batch_size, time_step, input_dim
        output, (h_n, c_n) = self.lstm(attention_output)
#         print(output.shape) # torch.Size([16, 1, 64]) batch_size, time_step, hidden_dim
        batch_size, timestep, hidden_dim = output.shape
        output = output.reshape(-1, hidden_dim)
        output = self.fc(output)
        output = output.reshape(timestep, batch_size, -1)
        return output[-1]


model = LSTM(input_dim, hidden_dim, num_layers, output_dim)  # 定义LSTM网络
loss_function = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 定义优化器

# 8.模型训练
for epoch in range(epochs):
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        optimizer.zero_grad()
        y_train_pred = model(x_train, x_train, x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

    # 模型验证
    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test_pred = model(x_test, x_test, x_test)
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), save_path)

print('Finished Training')

# 9.绘制结果
#修改部分
# 使用模型进行预测
model.eval()
predicted = []
actual = []
with torch.no_grad():
    for data in test_loader:
        x_test, y_test = data
        y_test_pred = model(x_test, x_test, x_test)
        predicted.extend(y_test_pred.numpy())
        actual.extend(y_test.numpy())

# 转换列表为numpy数组
predicted = np.array(predicted)
actual = np.array(actual)

# 绘制预测值和实际值
plt.figure(figsize=(10,5))
plt.plot(actual, label='Actual Temperature')
plt.plot(predicted, label='Predicted Temperature')
plt.title('Temperature Prediction Comparison')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()