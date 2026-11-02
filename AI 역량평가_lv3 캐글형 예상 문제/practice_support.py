from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parent
def csv(q,name): return pd.read_csv(ROOT/'data'/q/name)
def images(name):
 with np.load(ROOT/'data/q2'/name,allow_pickle=False) as z: return z['ids'].astype(str).copy(),z['images'].copy()
def save(frame,name):
 folder=ROOT/'제출물'; folder.mkdir(exist_ok=True); path=folder/name; frame.to_csv(path,index=False); print('저장:',path); return path
