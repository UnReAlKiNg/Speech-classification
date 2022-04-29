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

##读取训练数据
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

##编辑Dataset  
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
