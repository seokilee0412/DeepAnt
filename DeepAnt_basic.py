#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import array
import random
from random import randint
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# # 하이퍼 파라미터

# In[2]:


w = 2000                 # History window (number of time stamps taken into account) 
                         # i.e., filter(kernel) size       
p_w = 300                # Prediction window (number of time stampes required to be 
                         # predicted)
n_features = 1           # Univariate time series

kernel_size = 2          # Size of filter in conv layers
num_filt_1 = 32          # Number of filters in first conv layer
num_filt_2 = 32          # Number of filters in second conv layer
num_nrn_dl = 40          # Number of neurons in dense layer
num_nrn_ol = p_w         # Number of neurons in output layer

conv_strides = 1
pool_size_1 = 2          # Length of window of pooling layer 1
pool_size_2 = 2          # Length of window of pooling layer 2
pool_strides_1 = 2       # Stride of window of pooling layer 1
pool_strides_2 = 2       # Stride of window of pooling layer 2

epochs = 30
dropout_rate = 0.5       # Dropout rate in the fully connected layer
learning_rate = 2e-5  
anm_det_thr = 0.8        # Threshold for classifying anomaly (0.5~0.8)


# # Anomaly Detector

# In[3]:


def anomaly_detector(prediction_seq, ground_truth_seq):
    
    dist = np.linalg.norm(ground_truth_seq - prediction_seq)
    
    if (dist > anm_det_thr):
        return true  # anomaly
    else:
        return false # normal


# # Data 불러오기 / 임의 생성

# In[4]:


df_sine = pd.read_csv('https://raw.githubusercontent.com/swlee23/Deep-Learning-Time-Series-Anomaly-Detection/master/data/sinewave.csv')
# plt.plot(df_sine['sinewave'])
# plt.title('sinewave')
# plt.ylabel('value')
# plt.xlabel('time')
# plt.legend(['sinewave'], loc='upper right')
# plt.figure(figsize=(100,10))
# plt.show()
# df_sine.head()


# # 데이터 window_size로 나누기

# In[9]:


def split_sequence_to_dataset(sequence):
    X, y = list(), list()
    for i in range(len(sequence)):

        end_ix = i + w
        out_end_ix = end_ix + p_w
        
        if out_end_ix > len(sequence):
            break
            
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        
        X.append(seq_x)
        y.append(seq_y)
    
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], n_features)
    y = np.array(y)
    
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
#     print(X.shape)
    tensor_dataset = TensorDataset(X,y)
        
    return tensor_dataset
raw_seq = list(df_sine['sinewave'])


# In[10]:


train_dataset = split_sequence_to_dataset(raw_seq)

train_loader = DataLoader(train_dataset, shuffle = False, batch_size = 1, drop_last=False)


# # 모델  만들기

# In[11]:


class DeepAnt(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding=0):
        super(DeepAnt,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv1d_1 = nn.Conv1d(in_channels,out_channels,kernel_size = kernel_size, stride = stride, padding = padding)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(kernel_size = pool_size_1)
        
        self.conv1d_2 = nn.Conv1d(out_channels,in_channels * out_channels,kernel_size = kernel_size, stride = stride, padding = padding)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool1d(kernel_size = pool_size_1)
        
        self.dropout = nn.Dropout(p = dropout_rate)
        
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear( int((w/4 - 1) * num_filt_1),1024)
        self.relu_3 = nn.ReLU()
        self.output_dense = nn.Linear(1024,p_w)
    
    def forward(self,x):
        x = x.reshape(x.size(0),x.size(2),x.size(1))
        output = self.conv1d_1(x)
        output = self.relu_1(output)
        output = self.maxpool_1(output)
#         print(output.shape) # torch.Size([1, 32, 999])
        
        
        output = self.conv1d_2(output)
        output = self.relu_2(output)
        output = self.maxpool_2(output)
#         print(output.shape) #torch.Size([1, 32, 499])
        
        output = self.flatten(output)
#         print(output.shape)
#         print(int(w * (num_filt_1/4) * num_filt_1))
        output = self.dense_1(output)
        output = self.dropout(output)
        output = self.relu_3(output)
        
        output = self.output_dense(output)
        
        return output


# In[12]:


model = DeepAnt(in_channels = n_features, out_channels = num_filt_1, kernel_size = kernel_size, stride = conv_strides, padding=0)


# In[13]:


loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr = 1e-6 )


# # 학습

# In[14]:


def training(model,optimizer,loss_func,train_loader,epochs):
    loss_arr = []
    for i in range(epochs):
        
        optimizer.zero_grad()
        loss = 0
        for j, (x,y) in enumerate(train_loader):
            y = y.type(torch.int64)
            predict = model(x.float())
            loss += loss_func(predict,y) 

        loss.backward()
        optimizer.step()
        loss_arr.append(loss)

#   if i %10 == 0:
#     print("%d번째 loss:%d "%(i+1,loss))
    return model, loss_arr


# In[15]:


model = model.float()
model, loss_arr = training(model, optimizer, loss_func, train_loader, epochs)


# In[16]:


loss_arr


# In[17]:


plt.plot(loss_arr)
plt.show()


# # 테스트

# In[18]:


# Set number of test sequences 
n_test_seq = 1

# Split a univariate sequence into samples
def generate_test_batch(raw_seq, n_test_seq):
  # Sample a portion of the raw_seq randomly
    ran_ix = random.randint(0,len(raw_seq) - n_test_seq * w - n_test_seq * p_w)
    raw_test_seq = array(raw_seq[ran_ix:ran_ix + n_test_seq * w +  n_test_seq * p_w])
    batch_test_seq, batch_test_label = list(), list()
    ix = ran_ix
    for i in range(n_test_seq):
        # gather input and output parts of the pattern
        seq_x = raw_seq[ix : ix+w],
        seq_y = raw_seq[ix+w : ix+w+p_w]
        ix = ix+w+p_w
        batch_test_seq.append(seq_x)
        batch_test_label.append(seq_y)
        
        
    batch_test_seq = np.array(batch_test_seq)
    batch_test_seq = batch_test_seq.reshape(batch_test_seq.shape[0], batch_test_seq.shape[2], n_features)
    batch_test_label = np.array(batch_test_label)
    batch_test_label = batch_test_label.reshape(batch_test_label.shape[0], p_w)
    
    batch_test_seq = torch.from_numpy(batch_test_seq)
    batch_test_label = torch.from_numpy(batch_test_label)
    
    test_dataset = TensorDataset(batch_test_seq,batch_test_label)
        
    return test_dataset


# In[19]:


test_dataset = generate_test_batch(list(df_sine['sinewave']), n_test_seq)

test_loader = DataLoader(test_dataset, shuffle = False, batch_size = 1, drop_last=False)


# In[20]:


def testing(testdataloader,model_):
    model_.eval()
    loss = 0
    eval_loss = []
    predict_array = []
    with torch.no_grad():
        for i,(x,y) in enumerate(testdataloader):
#             x,y = Variable(x), Variable(y)
#             y = y.type(torch.int64)
            
            predict = model_(x.float())
            predict_array.append(predict.reshape(-1))
            loss += loss_func(predict,y)
            
    print('Loss : ', loss)
    
    return predict_array


# In[21]:


predicted_seq = testing(test_loader, model)
predicted_seq = np.array(predicted_seq[0])


# # 예측 결과 시각화

# In[22]:


raw_seq = list(df_sine['sinewave'])
endix = len(raw_seq) - w - p_w
input_seq = array(raw_seq[endix:endix+w])
target_seq = array(raw_seq[endix+w:endix+w+p_w]) 
input_seq = input_seq.reshape((1, w, n_features))


# In[23]:


in_seq = df_sine['sinewave'][endix:endix+w]
tar_seq = df_sine['sinewave'][endix+w:endix+w+p_w]
predicted_seq = predicted_seq.reshape((p_w))
d = {'time': df_sine['time'][endix+w:endix+w+p_w], 'values': predicted_seq}
df_sine_pre = pd.DataFrame(data=d)
pre_seq = df_sine_pre['values']

plt.plot(in_seq)
plt.plot(tar_seq)
plt.plot(pre_seq)
plt.title('sinewave prediction')
plt.ylabel('value')
plt.xlabel('time')
plt.legend(['input_seq', 'target_seq', 'pre_seq'], loc='upper right')
axes = plt.gca()
axes.set_xlim([endix,endix+w+p_w])
fig_predict = plt.figure(figsize=(100,10))
# fig_predict.savefig('predicted_sequence.png')
plt.show()


# # 모델 저장하기

# In[ ]:


#모델 저장하기
torch.save(model.state_dict(), 'Basic_DeepAnt.pt')

