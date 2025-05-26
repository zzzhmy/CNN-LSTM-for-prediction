import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.nn import MaxPool2d, Conv2d, Dropout, ReLU
from torch.utils.data import DataLoader, Dataset
import random

seed = 42
torch.manual_seed(seed)
random.seed(seed)
#准备数据集
#df=pd.read_csv("/data2/zmy/no_null_Residential_14.csv",parse_dates=["date"],index_col=[0])
df=pd.read_csv('/data2/zmy/lorenz_dataset-3.csv',parse_dates=['time'],index_col=[0])
print(df.shape)
train_data_size=round(len(df)*0.833)
test_data_size=round(len(df)*0.167)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

a=df[['y']]
plt.plot(a,'g') #读取到Open这一列，列名就是1
plt.ylabel("y_value")
plt.xlabel("times")
#plt.show()
plt.ylim([-40, 70])  # 设置y轴范围
plt.savefig('/data2/zmy/img/y_value.png')

sel_col=['y','x','z']
df=df[sel_col]

df_close_max=df['y'].max()
df_close_min=df['y'].min()
print("最高负荷=", df_close_max)
print("最低负荷=", df_close_min)


df=df.apply(lambda x:(x-min(x))/(max(x)-min(x))) #标准化到0~1之间
print(df)

total_len=df.shape[0]
print("df.shape=",df.shape) #数据的维度，几行几列
print("df_len=", total_len) #列数即数据长度

sequence=10
x=[]
y=[]

#有点像可能是滑动窗口
for i in range(total_len-sequence):

    x.append(np.array(df.iloc[i:(i+sequence),].values,dtype=np.float32)) #取从i开始的7行*24个值，
    y.append(np.array(df.iloc[(i+sequence),1],dtype=np.float32)) #7天预测出一天
print("shape of train data item  0:  \n", x[0].shape)
print("train label of item  0: \n", y[0])

print("\n序列化后的，也就是转换成3D张量后的数据形状：")
X = np.array(x)
Y = np.array(y)
Y = np.expand_dims(Y, 1)
print("X.shape =",X.shape)
print("Y.shape =",Y.shape)
x_tensor=torch.from_numpy(X)
y_tensor=torch.from_numpy(Y)

train_x = x_tensor[:int(0.83 * total_len)]
train_y = y_tensor[:int(0.83 * total_len)]


# 数据集前80%后的数据（20%）作为验证集
valid_x = x_tensor[int(0.83 * total_len):]
valid_y = y_tensor[int(0.83 * total_len):]




#把读取到的数据，人为的分为训练集和测试集
class Mydataset(Dataset):

    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        return x1, y1

    def __len__(self):
        return len(self.x)


#构建合适dataloader数据加载器来datalode的数据集
dataset_train = Mydataset(train_x, train_y)
dataset_valid = Mydataset(valid_x, valid_y)
#启动dataloader，关闭shuffle的打乱顺序
train_dataloader=DataLoader(dataset_train,batch_size=64)
valid_dataloader=DataLoader(dataset_valid,batch_size=64)
# print(train_dataloader)
# print(valid_dataloader)

class MAPELoss(torch.nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        '''
        Args:
        - y_pred: 预测值, size=[batch_size]
        - y_true: 目标值, size=[batch_size]
        Returns:
        - loss: MAPE值
        '''
        epsilon = 1e-8
        diff = torch.abs((y_true - y_pred) / torch.clamp(y_true, min=epsilon))
        loss_mape = torch.mean(diff)*100
        return loss_mape
    
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        '''
        Args:
        - y_pred: 预测值, size=[batch_size]
        - y_true: 目标值, size=[batch_size]
        Returns:
        - loss: RMSE值
        '''
        criteron = torch.nn.MSELoss()
        mse_loss = criteron(y_pred, y_true)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss

class cnn_lstm(nn.Module):
    def __init__(self,window_size,feature_number):
        super(cnn_lstm, self).__init__()
        self.window_size=window_size
        self.feature_number=feature_number
        self.conv1 = Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.maxpooling1 = MaxPool2d(3, stride=1,padding=1)
        self.dropout1 = Dropout(0.3)
        self.lstm1 = nn.LSTM(input_size=64 * feature_number, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=64, out_features=32)
        self.relu2 = nn.ReLU()
        self.head = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):

            x = x.reshape([x.shape[0], 1, self.window_size, self.feature_number])
            # x = x.transpose(-1, -2)
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.maxpooling1(x)
            x = self.dropout1(x)

            x = x.reshape([x.shape[0], self.window_size, -1])
            # x = x.transpose(-1, -2)  #
            x, (h, c) = self.lstm1(x)
            x, (h, c) = self.lstm2(x)
            #x = x[:, -1, :]  # 最后一个LSTM只要窗口中最后24个特征的输出
            # 隐藏层的输出，就是全连接层的输入
            x = self.fc(x)
            x = self.relu2(x)
            x = self.head(x)
            x = x[:, -1, :]
            #print('x.shape',x.shape)
            return x

#创建实例化网络模型
cnn_lstm=cnn_lstm(window_size=10,feature_number=3)
print(cnn_lstm)

#定义损失函数
loss_fn=RMSELoss()
mape_loss_func=MAPELoss()
r2_loss_func=nn.MSELoss()

#定义优化器
learning_rate=0.01
opitmizer=torch.optim.Adam(cnn_lstm.parameters(),learning_rate)

#设置训练网络参数
total_train_step=0
total_valid_step=0

#训练论数
epoch=70
hist = np.zeros(epoch)
mape_hist = np.zeros(epoch)
r2_hist=np.zeros(epoch)

for i in range(epoch):
    #print("______第{}轮训练开始________".format((i + 1)))
    y_train_pred=cnn_lstm(train_x)
    loss=loss_fn(y_train_pred,train_y)
    mape_loss=mape_loss_func(y_train_pred,train_y)
    r2_loss=1-r2_loss_func(y_train_pred,train_y)/torch.var(train_y)

    if i % 10 == 0 and i != 0:  # 每训练十次，打印一次均方差
        print("Epoch ", i, "RMSE: ", loss.item(),"MAPE:",mape_loss.item(),"r2:",r2_loss.item())
    hist[i] = loss.item()
    mape_hist[i]=mape_loss.item()
    r2_hist[i]=r2_loss.item()
    
    #优化器优化模型
    opitmizer.zero_grad()
    loss.backward()
    opitmizer.step()

y_train_pred=cnn_lstm(train_x)
loss_fn(y_train_pred,train_y).item()
mape_loss_func(y_train_pred,train_y).item()
r2_loss_func(y_train_pred,train_y).item()

    # total_train_step = total_train_step + 1
    # if total_train_step % 100 == 0:
    #     print("训练次数：{}，loss:{}".format(total_train_step, loss.item()))
#显示batch accuracy的历史数据
fig = plt.figure() # 创建新的Figure对象
plt.plot(hist, "r")
plt.grid()
plt.xlabel("times")
plt.ylabel("RMSE_loss")
plt.title("RMSE_loss", fontsize = 12)
#plt.show()
plt.savefig('/data2/zmy/img/cnn_lstm_loss.png')

fig = plt.figure() # 创建新的Figure对象
plt.plot(mape_hist, "r")
plt.grid()
plt.xlabel("times")
plt.ylabel("MAPE_loss")
plt.title("MAPE_loss", fontsize = 12)
#plt.show()
plt.savefig('/data2/zmy/img/cnn_lstm_mapeloss.png')
fig = plt.figure() # 创建新的Figure对象
plt.plot(r2_hist, "r")
plt.grid()
plt.xlabel("times")
plt.ylabel("r2")
plt.title("r2", fontsize = 12)
#plt.show()
plt.savefig('/data2/zmy/img/cnn_lstm_r2.png')

# y_test_pred=cnn_lstm(valid_x)
# loss_fn(y_test_pred,valid_y)

cnn_lstm.eval()
with torch.no_grad():
    # 使用验证集进行预测
    data_loader = valid_dataloader
    # 存放测试序列的预测结果
    predicts = []
    # 存放测试序列的实际发生的结果
    labels = []
    for idx, (x, label) in enumerate(data_loader):
        if (x.shape[0] != 64):
            continue
        # 对测试集样本进行批量预测，把结果保存到predict Tensor中
        # 开环预测：即每一次序列预测与前后的序列无关。
        predict= cnn_lstm(x)
        # 把保存在tensor中的批量预测结果转换成list
        predicts.extend(predict.data.squeeze(1).tolist())

        # 把保存在tensor中的批量标签转换成list
        labels.extend(label.data.squeeze(1).tolist())

    predicts = np.array(predicts)
    labels = np.array(labels)
    print(predicts.shape) #输出shape为(7680,24)
    print(labels.shape) #输出shape为(7680,)

    predicts_unnormalized = df_close_min + (df_close_max - df_close_min) * predicts
    labels_unnormalized = df_close_min + (df_close_max - df_close_min) * labels

    print("shape:", predicts_unnormalized.shape)
    #print("正则标准化后的预测数据：\n", predicts)
    print("")
    print("正则标准化前的预测数据：\n", predicts_unnormalized)

    fig = plt.figure() # 创建新的Figure对象
    plt.grid()
    plt.plot(predicts_unnormalized,"r",linestyle="--",label="pred_y")
    plt.plot(labels_unnormalized,  "b",label="real_y")
    plt.xlabel("times")
    plt.ylabel("compare")
    #plt.show()
    plt.ylim([-40, 70])  # 设置y轴范围
    plt.savefig('/data2/zmy/img/cnn_lstm_pre_y.png')

# m, n = np.max(load), np.min(load)
# load = (load - n) / (m - n)                      # 当前值-min/（max-min），我不太理解这里的load处理是什么意思！很懵逼
# for i in range(0, len(dataset) - 24 - num, num):  # num表示预测的步长，num=4表面预测接下来四个时刻，i表示时间序列维度
#     train_seq = []
#     train_label = []
#     for j in range(i, i + 24):   # j是第几步，代表着一个长度24的时间滑块在总时间轴滑动
#         x = [load[j]]   # 把负荷load值传给x
#         for c in range(2, 8):           # 为何是2-8呢，我看data里只有7列呀？
#             x.append(dataset[j][c])     # 把数据处理好放到x，再把x的值传到train_seq
#             train_seq.append(x)
#     for j in range(i + 24, i + 24 + num):  # 标签值num个，最后得到的预测值也是这么多个
#             train_label.append(load[j])           # num个标签值，对应预测值进行对比
#     train_seq = torch.FloatTensor(train_seq)                  # 类型转换：转变成Tensor
#     train_label = torch.FloatTensor(train_label).view(-1)     # 把原有张量变成一维结构
#     seq.append((train_seq, train_label))  # 是不是相当于有len（dataset）-24-num个train_seq片段呀