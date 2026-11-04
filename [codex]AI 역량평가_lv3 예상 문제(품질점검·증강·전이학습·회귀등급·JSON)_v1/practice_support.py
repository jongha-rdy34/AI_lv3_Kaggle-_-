from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
ROOT=Path(__file__).resolve().parent
CLASSES=['circle','square','triangle','cross']

def load_data():
    meta=pd.read_csv(ROOT/'data/train_metadata.csv')
    with np.load(ROOT/'data/train_images.npz',allow_pickle=False) as z: images=z['images'].copy()
    return meta,images

def ready_splits():
    return pd.read_csv(ROOT/'data/ready_splits.csv')

def test_images():
    with np.load(ROOT/'data/test_images.npz',allow_pickle=False) as z: return z['ids'].astype(str).copy(),z['images'].copy()

def load_q4():
    frame=pd.read_csv(ROOT/'data/q4_scores.csv')
    thresholds=json.loads((ROOT/'data/q4_thresholds.json').read_text(encoding='utf8'))['thresholds']
    return frame,np.asarray(thresholds,dtype=float)

class Backbone(nn.Module):
    out_channels=64
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.block2=nn.Sequential(nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.block3=nn.Sequential(nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.block4=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU())
    def forward(self,x): return self.block4(self.block3(self.block2(self.block1(x))))

def load_backbone():
    net=Backbone()
    net.load_state_dict(torch.load(ROOT/'assets/pretrained_backbone.pt',map_location='cpu',weights_only=True))
    return net

class ReadyDataset(Dataset):
    # Q3·Q4 독립 진행용 기본 Dataset. Q2에서는 별도로 Dataset을 작성합니다.
    def __init__(self,rows,images,training=False,regression=False):
        self.rows=rows.reset_index(drop=True);self.images=images;self.training=training;self.regression=regression
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows.iloc[i]; image=self.images[int(row['image_index'])]
        if self.training and random.random()<.5: image=np.flip(image,axis=1).copy()
        x=torch.from_numpy(np.array(image,copy=True,order="C")).permute(2,0,1).float()/255.0
        y=torch.tensor(float(row['score'])/100,dtype=torch.float32) if self.regression else torch.tensor(CLASSES.index(row['label']),dtype=torch.long)
        return x,y

def ready_loaders(images,rows,regression=False,batch_size=64):
    train=rows[rows['split'].eq('train')].reset_index(drop=True); valid=rows[rows['split'].eq('valid')].reset_index(drop=True)
    train_ds=ReadyDataset(train,images,training=True,regression=regression); valid_ds=ReadyDataset(valid,images,training=False,regression=regression)
    # 분류는 클래스 역빈도 샘플링, 회귀는 일반 셔플을 사용합니다.
    if regression:
        tr=DataLoader(train_ds,batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(42),num_workers=0)
    else:
        counts=train['label'].value_counts(); weights=train['label'].map(lambda y:1.0/counts[y]).to_numpy(dtype=float,copy=True)
        sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),len(weights),replacement=True,generator=torch.Generator().manual_seed(42))
        tr=DataLoader(train_ds,batch_size=batch_size,sampler=sampler,num_workers=0)
    return tr,DataLoader(valid_ds,batch_size=batch_size,shuffle=False,num_workers=0)

def save_submission(frame,sample_file,name):
    sample=pd.read_csv(ROOT/'data'/sample_file)
    if list(frame.columns)!=list(sample.columns): raise ValueError('제출 열/순서가 샘플과 다릅니다.')
    if frame['id'].duplicated().any() or frame['id'].tolist()!=sample['id'].tolist(): raise ValueError('id 중복이나 순서 오류')
    if frame.isna().any().any(): raise ValueError('제출 데이터에 결측이 있습니다.')
    path=ROOT/'제출물'/name;path.parent.mkdir(exist_ok=True);frame.to_csv(path,index=False);print('저장:',path);return path

def validate_json_schema(item):
    # JSON의 구문·키·자료형·허용값만 점검하며 내용이 사실에 근거하는지는 판단하지 않습니다.
    keys={'id','model','issues','duration_minutes','warranty','repair_completed'}
    if not isinstance(item,dict) or set(item)!=keys: return False
    if not isinstance(item['id'],str): return False
    if item['model'] is not None and not isinstance(item['model'],str): return False
    if not isinstance(item['issues'],list) or any(not isinstance(x,str) or x not in {'battery','fan','storage','display'} for x in item['issues']): return False
    if item['duration_minutes'] is not None and (type(item['duration_minutes']) is not int or item['duration_minutes']<0): return False
    for k in ['warranty','repair_completed']:
        if item[k] is not None and type(item[k]) is not bool: return False
    return True
