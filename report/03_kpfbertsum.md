## KPFBERTSUM
- 진행중입니다

### [KPFBERTSUM / kpfbertsum](https://github.com/KPFBERT/kpfbertsum)
### [Detail Document](https://github.com/KPFBERT/kpfbertsum/blob/main/kpfbert_summary.ipynb)

### Test Summary
- 문서 및 코드 분석 중
- AIHub 데이터셋 다운로드
```
https://aihub.or.kr/aidata/8054
```
- 테스트 코드 작성 중

### TODO
- 코드 분석 및 테스트 코드 작성
- 테스트 결과 확인

---

### Environment
```
- Windows 10 Pro
- Python 3.9.1
- 내장 그래픽카드 : Intel UHD Graphics 630
```

### Sample Code: In progress
- 샘플 데이터 분석 및 코드 수정 필요
- 현재 샘플 데이터와 코드에서의 포맷이 일치하지 않아 실행 오류 발생: `KeyError: 'article_original'`
```python
import math
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from torch.nn.init import xavier_uniform_

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import kss

# jupiter notebook issues
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

RANDOM_SEED = 42

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

pl.seed_everything(RANDOM_SEED)

MAX_TOKEN_COUNT = 512
N_EPOCHS = 10
BATCH_SIZE = 4

#DATA_TRAIN_PATH = 'data/train_original.json'
DATA_TRAIN_PATH = './train_original.json'
df = pd.read_json(DATA_TRAIN_PATH)
df = df.dropna()
print ( len(df) ) #, len(val_df)

#DATA_TEST_PATH = 'data/vaild_original.json'
DATA_TEST_PATH = './valid_original.json'
test_df = pd.read_json(DATA_TEST_PATH)
test_df = test_df.dropna()
print ( len(test_df) )

train_df, val_df = train_test_split(df, test_size=0.05)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print ( train_df.shape, val_df.shape, test_df.shape )

# test setting all data downsize
downsize = 2000
train_df = train_df[:downsize]
test_df = test_df[:downsize//10]
val_df = val_df[:downsize//10]
print ( train_df.shape, test_df.shape, val_df.shape )

i = 8
print('===== 본    문 =====')
for idx, str in enumerate(train_df['article_original'][i]):
    print(idx,':',str)
print('===== 요약정답 =====')
print(train_df['extractive'][i])
print('===== 추출본문 =====')
print('1 :', train_df['article_original'][i][train_df['extractive'][i][0]])
print('2 :', train_df['article_original'][i][train_df['extractive'][i][1]])
print('3 :', train_df['article_original'][i][train_df['extractive'][i][2]])
print('===== 생성본문 =====')
print(train_df['abstractive'][i])

print ( test_df.head(1) )
```