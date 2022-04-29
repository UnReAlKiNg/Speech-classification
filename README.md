# Speech-classification

这篇文章旨在帮助音频分类初学者更好地了解音频分类的相关内容 \
数据集：https://www.kaggle.com/competitions/speech-command-classification/data  

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
这里建议先创建文件夹“speech-classification”，将数据集压缩包下载好之后解压到文件夹“speech-classification/audio_files/”下。同时，将文件“speech_hw.ipynb”和"sample_submission.csv"也放到目录“speech-classification”下。  
具体格式为：  
speech-classification  
｜  
｜-audio_files(所有音频文件)  
｜-speech_hw.ipynb  
｜-sample_submission.csv  

准备好之后可以通过(我是macOS系统，windows可以自行查找一下相关cmd命令)
```zsh
!ls
```
进行查看

