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
