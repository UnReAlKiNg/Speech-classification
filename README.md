# Speech-classification

这篇文章旨在帮助音频分类初学者更好地了解音频分类的相关内容 \
数据集：https://www.kaggle.com/competitions/speech-command-classification/data  
代码的具体运行结果可以参考“speech.ipynb”  

## 预先准备
首先我们引入一些必要的python库  
```python
%matplotlib inline
import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import model_selection
from sklearn import preprocessing
import IPython.display as ipd
import cv2
```
这里建议先创建文件夹“speech-classification”，将数据集压缩包下载好之后解压到文件夹“speech-classification/audio_files/”下。同时，将文件“speech.ipynb”，“train.csv”和"sample_submission.csv"也放到目录“speech-classification”下。  
具体格式为：  
speech-classification  
｜  
｜-audio_files(所有音频文件)  
｜-speech.ipynb  
｜-sample_submission.csv  
｜-train.csv  

准备好之后可以通过(我是macOS系统，windows可以自行查找一下相关cmd命令)
```zsh
!ls
```
进行查看

## 读取训练数据
```python
train_file = "train.csv"
audio_dir = "audio_files/"

meta_data = pd.read_csv(train_file)
meta_data.head()
```
查看数据大小
```python
data_size = meta_data.shape
data_size
```
可以看到“audio_files”里100000+的数据里，74080为训练数据，剩下30000+为需要做出预测的数据  
接下来我们开始划分数据集，这里采用80%训练+20%验证的比例  
```python
x = list(meta_data.loc[:,"file_name"])
y = list(meta_data.loc[:, "target"])

print(len(x),len(y))
target_set = set(y)
class_dic = dict()

for i,j in enumerate(target_set):
    class_dic[j] = i

print(class_dic)
x_dir = []
y_dir = []
ind = 0

while ind < len(x):
    file_path = audio_dir + x[ind]
    class_num = class_dic[y[ind]]
    x_dir.append(file_path)
    y_dir.append(class_num)

    ind = ind + 1  
    #if ind%(len(x)/10) == 0:
    #    print('process = {}%'.format(ind*10/(len(x)/10)))
    #    print("x_dir = {}, y_dir = {}".format(len(x_dir),len(y_dir)))    

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_dir, y_dir, test_size=0.2, stratify=y_dir)
print("x train:{0}\ny train:{1}\nx test:{2}\ny test:{3}".format(len(x_train),
                                                                len(y_train), 
                                                                len(x_test), 
                                                                len(y_test)))
print("x_train = \n", x_train[0:10])
print("y_train = \n", y_train[0:10])                                                    
```
这里会生成一个标签-序号的字典（例如{'bird': 0, 'zero': 1,...},每次运行字典会变化），一共35个类别  

## 编辑Dataset  
首先还是引入必要的库
```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
```
Dataset的具体细节如下:  
```python
class SpeechDataset(Dataset):
    def __init__(self, data_path, target=None, is_test=False, augmentation=None):
        super().__init__()
        self.data_path = data_path
        self.target = target
        self.is_test = is_test
        self.augmentation = augmentation
        self.duration = 1000
        self.sr = 16000
        self.n_fft = 1024
        self.hop_length = None
        self.n_mels = 64
        self.top_db = 80

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self,idx):
        file_path = self.data_path[idx]
        class_id = self.target[idx]

        samples,sr = torchaudio.load(file_path)
        samples = self._pad_trunc(samples,self.sr)

        spect = torchaudio.transforms.MelSpectrogram(
            self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )(samples)

        spect = torchaudio.transforms.AmplitudeToDB(top_db=self.top_db)(spect)
        spect = self.rechannel(spect,self.sr,3)

        return spect, self.target[idx]
        

    def _pad_trunc(self, samples, sr):
        num_rows, signal_len = samples.shape
        max_len = sr // 1000 * self.duration

        if (signal_len > max_len):
            # Truncate the signal to the given length
            samples = samples[:, max_len]

        elif (signal_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - signal_len)
            pad_end_len = max_len - signal_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((1, pad_begin_len))
            pad_end = torch.zeros((1, pad_end_len))

            samples = torch.cat((pad_begin, samples, pad_end), 1)

        return samples

    def rechannel(self, spect, sr, num_channel):
        if (spect.shape[0] == num_channel):
            # Nothing to do
            return spect

        if (num_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            spect = spect[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            spect = torch.cat([spect, spect, spect])

        return spect

    def _time_shift(self, samples, sr, shift_limit):
        _, sig_len = samples.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return samples.roll(shift_amt)
```
核心思想是把音频统一到一个时间长度，并把torchaudio.transforms.MelSpectrogram生成的梅尔频谱(单色图)转换成三色图，好放入网络。或者也可以不转换成三色图，修改网络第一层的参数也可以，例如对于resnet系列，可以使用  
```python
import torch.nn as nn
import torchvision
model = torchvision.models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
```
这里使用torchaudio而不是librosa的原因是，librosa效率真的很低，同样设置(程序参数、模型、电脑配置(MacBook Pro，M1))，torchaudio可以只用50%左右的cpu，实现8分钟一个epoch，而librosa却要占用近90%的cpu，同时还要30分钟一个epoch，效率差距显而易见

## 创建数据集  
定义一些参数：  
```python
config = {
    'batch_size': 64,
    'num_workers': 0,
    'epochs': 10,
    'device': 'cpu'
}
```
以及生成数据集：  
```python
train_dataset = SpeechDataset(
    data_path = x_train,
    target = y_train,
    is_test = False
)
valid_dataset = SpeechDataset(
    data_path = x_test,
    target = y_test,
    is_test = False
)
train_loader = DataLoader(
    train_dataset,
    batch_size = config['batch_size'],
    shuffle = True,
    num_workers = config['num_workers'],
    drop_last = True
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size = config['batch_size'],
    shuffle = False,
    num_workers = config['num_workers'],
    drop_last = False
)
```

## 设置模型  
这里采用的是resnet18，因为我们的训练数据规模较小，使用大的网络(如resnet34或者resnet50)会出现过拟合。优化器和损失函数还是经典的Adam+CrossEntropyLoss组合。  
```python
model = torchvision.models.resnet18(num_classes=35)
#model = torchvision.models.mobilenet_v3_small(num_classes=35)
#model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model.to(config['device'])

# 获取优化方法
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, betas=(0.9,0.99), eps=1e-6, weight_decay=5e-4)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-2)
# 获取学习率衰减函数
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 4], gamma=0.3)
# 获取损失函数
loss = torch.nn.CrossEntropyLoss()
device = 'cpu'
```

## 定义训练和验证函数
```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def validation(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Val Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
```

## 开始训练！  
```python
epochs = 30
for t in tqdm(range(epochs)):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_loader, model, loss, optimizer)
    validation(valid_loader, model, loss)

print("Done!")
```

## 读取待预测的文件  
```python
submission = pd.read_csv("sample_submission.csv")
test_filename = list(submission.loc[:,"file_name"])
test_dir = []
for item in test_filename:
  ans = audio_dir + item
  test_dir.append(ans)

test_dir[0:10]
```

## 预测  
首先将Dataset里面使用的函数搬过来，对待预测文件进行相同处理(注意使用self.xxx的地方要进行修改)。同时get_keys()函数是用来通过序号反找对应标签(因为标签-序号是单值对应，所以不会重复)  
```python
def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

result = []
for fpath in tqdm(test_dir):   
    samples, sr = torchaudio.load(fpath)
    samples = pad_trunc(samples,16000)

    spect = torchaudio.transforms.MelSpectrogram(
        sample_rate = 16000,
        n_fft=1024,
        hop_length=None,
        n_mels=64
    )(samples)

    spect = torchaudio.transforms.AmplitudeToDB(top_db=80)(spect)
    spect = rechannel(spect,16000,3)
    spect = spect[np.newaxis,:]

    
    output = model(spect)
    ans = torch.nn.functional.softmax(output)
    ans = ans.data.cpu().numpy()
    lab = np.argsort(ans)[0][-1]
    result.append(lab)
    
res1 = []
for elem in result:
    tg = get_keys(class_dic, elem)
    res1.append(tg[0])    

submission["target"] = res1
submission.to_csv("new_submission.csv", index=None)
submission.head()
```

## Well done!
