"""모의평가 종료 후에만 실행하는 로컬 검증 도구. 공식 채점기나 실제 성적이 아닙니다."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,mean_squared_error,cohen_kappa_score
ROOT=Path(__file__).resolve().parent.parent
VERSION="V1"
sys.path.insert(0,str(ROOT))
from practice_support import CLASSES,validate_json_schema

def aligned(q):
    submission=ROOT/'제출물'/VERSION/f'{q}_submission.csv'
    if not submission.exists(): return None
    sample=pd.read_csv(ROOT/'data'/f'{q}_sample_submission.csv',dtype={'id':str})
    user=pd.read_csv(submission,dtype={'id':str})
    truth=pd.read_csv(ROOT/'정답_채점용'/f'{q}_truth.csv',dtype={'id':str})
    if list(user.columns)!=list(sample.columns): raise ValueError(f'{q}: 열 이름·순서 오류')
    if user.id.duplicated().any() or user.id.tolist()!=sample.id.tolist(): raise ValueError(f'{q}: ID 중복 또는 순서 오류')
    if user.isna().any().any(): raise ValueError(f'{q}: 결측값')
    if not truth.id.is_unique or set(truth.id)!=set(user.id): raise ValueError(f'{q}: ID 누락')
    return truth.merge(user,on='id',how='left',validate='one_to_one')

def score_q3():
    d=aligned('q3')
    if d is None:return None
    if not d.label.isin(CLASSES).all(): raise ValueError('q3: 허용되지 않은 레이블')
    return {'metric':'macro_f1','value':float(f1_score(d.truth,d.label,labels=CLASSES,average='macro',zero_division=0))}

def score_q4():
    d=aligned('q4')
    if d is None:return None
    values=d.score.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or (values>100).any(): raise ValueError('q4: 0~100의 유한한 수여야 합니다')
    threshold=np.asarray(json.loads((ROOT/'data/q4_thresholds.json').read_text(encoding='utf8'))['thresholds'])
    grades=np.digitize(values,threshold); target=np.digitize(d.truth.to_numpy(dtype=float),threshold)
    return {'rmse':float(np.sqrt(mean_squared_error(d.truth,values))),'qwk':float(cohen_kappa_score(target,grades,labels=[0,1,2,3,4],weights='quadratic'))}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['V1','V2','V3'],default='V1');VERSION=parser.parse_args().version
    for name,fn in [('q3',score_q3),('q4',score_q4)]:
        result=fn();print(name,':',json.dumps(result,ensure_ascii=False) if result is not None else '제출 파일 없음')
    gold=json.loads((ROOT/'정답_채점용/q5_예시정답.json').read_text(encoding='utf8'))
    print('q5: 프롬프트 문장·수작업 결과는 스스로 검토하세요. 제공 예시 JSON 스키마 유효:',all(validate_json_schema(x) for x in gold))
    print('Q1·Q2 및 풀이 과정·의미 충실도는 사람이 검토해야 합니다.')
