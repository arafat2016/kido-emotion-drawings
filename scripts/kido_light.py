# ============================================================================
#  KIDO — LIGHT & FAST  (single model, streaming, low RAM, ~3-5 min on T4)
#  ViT-B/16 frozen features -> linear classifier, dedup + leak-checked.
#  Run: Runtime -> T4 GPU, paste, run. Paste the RESULT block back.
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","-q","install","torch","torchvision",
                "huggingface_hub","scikit-learn","numpy","pillow","tqdm"], check=False)
import os, glob, time, hashlib, zipfile, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

import torch, torch.nn as nn
dev="cuda" if torch.cuda.is_available() else "cpu"
log(f"device: {dev}")

# ---- data (download+extract once, read from disk) ----
from huggingface_hub import hf_hub_download
zp=hf_hub_download(repo_id="serdarciftci/KIDO",filename="Dataset.zip",repo_type="dataset")
EX="/content/kido_extracted"
if not os.path.exists(EX):
    with zipfile.ZipFile(zp) as z: z.extractall(EX)
jpgs=sorted(glob.glob(f"{EX}/**/*.jpg",recursive=True)+glob.glob(f"{EX}/**/*.jpeg",recursive=True)+glob.glob(f"{EX}/**/*.png",recursive=True))
log(f"{len(jpgs)} files")

# ---- parse label + hash (tiny reads), then dedup ----
paths=[]; ys=[]; hh=[]
for p in tqdm(jpgs,desc="index"):
    e=os.path.splitext(os.path.basename(p))[0].split("-")[-1].upper()
    if e=="H": lbl=1
    elif e=="S": lbl=0
    else: continue
    try: im=Image.open(p).convert("L").resize((64,64))
    except: continue
    paths.append(p); ys.append(lbl); hh.append(hashlib.md5(np.asarray(im).tobytes()).hexdigest())
y=np.array(ys); hh=np.array(hh)
seen=set(); keep=[]
for i,h in enumerate(hh):
    if h not in seen: seen.add(h); keep.append(i)
keep=np.array(keep); paths=[paths[i] for i in keep]; y=y[keep]; hh=hh[keep]
log(f"unique: {len(y)} | happy={int((y==1).sum())} sad={int((y==0).sum())}")

# ---- split (leak-checked by hash) ----
from sklearn.model_selection import GroupShuffleSplit
idx=np.arange(len(y))
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=0).split(idx,y,hh))
assert len(set(hh[tr])&set(hh[te]))==0

# ---- ViT frozen features (streaming, batched) ----
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import Dataset,DataLoader
tf=T.Compose([T.Resize(224),T.CenterCrop(224),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
class DS(Dataset):
    def __len__(s): return len(paths)
    def __getitem__(s,i): return tf(Image.open(paths[i]).convert("RGB")), i
net=models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1); net.heads=nn.Identity(); net.eval().to(dev)
X=np.zeros((len(paths),768),np.float32)
with torch.no_grad():
    for xb,ix in tqdm(DataLoader(DS(),batch_size=64,num_workers=2),desc="ViT feats"):
        X[ix.numpy()]=net(xb.to(dev)).cpu().numpy()

# ---- train + evaluate ----
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score,confusion_matrix
clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced")).fit(X[tr],y[tr])
pp=clf.predict(X[te])
print("\n"+"="*40)
print(" RESULT (ViT-B/16 frozen, held-out test)")
print("="*40)
print(f"  Accuracy: {accuracy_score(y[te],pp):.4f}")
print(f"  Macro F1: {f1_score(y[te],pp,average='macro'):.4f}")
print(f"  Kappa:    {cohen_kappa_score(y[te],pp):.4f}")
print(f"  n_test:   {len(te)}")
print("  Confusion [rows=true sad/happy]:"); print(confusion_matrix(y[te],pp))
print("="*40)
