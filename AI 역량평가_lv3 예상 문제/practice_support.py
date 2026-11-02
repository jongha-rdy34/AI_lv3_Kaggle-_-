from pathlib import Path
import json, numpy as np, pandas as pd, torch
ROOT=Path(__file__).resolve().parent
def load_q1(): return tuple(pd.read_csv(ROOT/'data'/f) for f in ['products.csv','events.csv','labels.csv'])
def load_q2():
 with np.load(ROOT/'data/ready.npz') as z: return {k:z[k].copy() for k in z.files}
def load_q3():
 j=json.loads((ROOT/'data/search.json').read_text(encoding='utf8'))
 with np.load(ROOT/'data/search.npz') as z: return j['documents'],j['queries'],z['doc_vectors'].copy(),z['query_vectors'].copy()
def rmse(y,p):
 a=np.asarray(y).reshape(-1); b=np.asarray(p).reshape(-1)
 if a.shape!=b.shape or not np.isfinite(b).all(): raise ValueError('shape 또는 유한값 확인')
 return float(np.sqrt(np.mean((a-b)**2)))
class MockTokenizer:
 pad_token_id=0; eos_token_id=1
 def __call__(self,text,return_tensors='pt',truncation=True,max_length=128):
  x=torch.tensor([[2+ord(c)%120 for c in text][-max_length:]],dtype=torch.long); return {'input_ids':x,'attention_mask':torch.ones_like(x)}
 def decode(self,ids,skip_special_tokens=True): return '테스트 통과' if len(ids) else ''
class MockCausalLM:
 device=torch.device('cpu')
 def eval(self): return self
 def generate(self,input_ids,attention_mask,max_new_tokens,do_sample,pad_token_id):
  assert not do_sample and max_new_tokens>0
  return torch.cat([input_ids,torch.tensor([[7,8]],device=input_ids.device)],dim=1)
