from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,log_loss,accuracy_score
ROOT=Path(__file__).resolve().parent.parent

def score(q):
 names={'q1':['id','prediction'],'q2':['id']+[f'p{i}' for i in range(10)],'q3':['id','label']}
 sub=pd.read_csv(ROOT/'제출물'/f'{q}_submission.csv',dtype={'id':str}); truth=pd.read_csv(ROOT/'정답_채점용'/f'{q}_truth.csv',dtype={'id':str})
 if list(sub.columns)!=names[q]: raise ValueError(f'{q}: 열 이름 또는 순서 오류: {list(sub.columns)}')
 if not sub.id.is_unique or len(sub)!=len(truth) or set(sub.id)!=set(truth.id): raise ValueError(f'{q}: ID 중복·누락 또는 행 수 오류')
 df=truth.merge(sub,on='id',how='left',validate='one_to_one')
 if df.isna().any().any(): raise ValueError(f'{q}: 결측값')
 if q=='q1':
  v=df.prediction.to_numpy(dtype=float)
  if not np.isfinite(v).all() or (v<0).any() or (v>1).any(): raise ValueError('q1: 유한한 0~1 예측 필요')
  return {'metric':'rmse','value':float(mean_squared_error(df.truth,v)**.5)}
 if q=='q2':
  v=df[[f'p{i}' for i in range(10)]].to_numpy(dtype=float)
  if not np.isfinite(v).all() or (v<0).any() or (v>1).any() or not np.allclose(v.sum(axis=1),1,atol=1e-5): raise ValueError('q2: 유한한 확률과 행별 합 1 필요')
  v=v/v.sum(axis=1,keepdims=True)  # CSV 저장 시 발생한 미세한 반올림만 보정
  return {'metric':'log_loss','value':float(log_loss(df.truth.astype(int),v,labels=list(range(10))))}
 if not df.label.isin(['fan','battery','upgrade']).all(): raise ValueError('q3: 잘못된 라벨')
 return {'metric':'accuracy','value':float(accuracy_score(df.truth,df.label))}

if __name__=='__main__':
 for q in ['q1','q2','q3']:
  path=ROOT/'제출물'/f'{q}_submission.csv'
  if not path.exists(): print(q,': 제출 CSV 없음'); continue
  print(q,json.dumps(score(q),ensure_ascii=False))
