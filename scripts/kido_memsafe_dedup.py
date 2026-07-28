# ============================================================================
#  KIDO — MEMORY-SAFE fast frozen suite (won't crash Colab RAM)
#  Key fix: never holds all images in RAM. It streams each image ->
#  extracts features -> discards the image. Only small feature vectors
#  (2048/1280/768 floats) are kept. Runs 3 architectures x 5 seeds.
#  RUN: Runtime -> T4 GPU. ~5-10 min after the one-time download.
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","-q","install","torch","torchvision",
                "huggingface_hub","scikit-learn","numpy","pillow","tqdm","scipy"], check=False)

import os, glob, time, hashlib, zipfile, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

import torch, torch.nn as nn
dev="cuda" if torch.cuda.is_available() else "cpu"
log(f"device: {dev}" + ("" if dev=="cuda" else "  (no GPU; slower)"))
SEEDS=[0,1,2,3,4]

# ---- 1. download + extract once ----
from huggingface_hub import hf_hub_download
zip_path=hf_hub_download(repo_id="serdarciftci/KIDO",filename="Dataset.zip",repo_type="dataset")
EXDIR="/content/kido_extracted"
if not os.path.exists(EXDIR):
    log("extracting..."); 
    with zipfile.ZipFile(zip_path) as z: z.extractall(EXDIR)
log("data ready.")
jpgs=sorted(glob.glob(f"{EXDIR}/**/*.jpg",recursive=True)+glob.glob(f"{EXDIR}/**/*.jpeg",recursive=True)+glob.glob(f"{EXDIR}/**/*.png",recursive=True))
log(f"found {len(jpgs)} files")

# ---- 2. parse labels + hashes WITHOUT keeping images (low RAM) ----
paths=[]; ys=[]; hashes=[]; skip=0
log("indexing labels + hashes (streaming, low memory)...")
for p in tqdm(jpgs):
    stem=os.path.splitext(os.path.basename(p))[0]; parts=stem.split("-")
    if len(parts)<5: skip+=1; continue
    emo=parts[-1].upper()
    if emo=="H": lbl=1
    elif emo=="S": lbl=0
    else: skip+=1; continue
    try:
        im=Image.open(p).convert("L").resize((64,64))   # tiny, for hash only
        h=hashlib.md5(np.asarray(im).tobytes()).hexdigest()
    except: skip+=1; continue
    paths.append(p); ys.append(lbl); hashes.append(h)
y=np.array(ys); hashes=np.array(hashes)
dup=int(len(hashes)-len(set(hashes.tolist())))
log(f"before dedup: {len(y)} images, {dup} duplicates")

# ---- DEDUPLICATE: keep only the first occurrence of each unique image ----
seen=set(); keep=[]
for i,h in enumerate(hashes):
    if h not in seen:
        seen.add(h); keep.append(i)
keep=np.array(keep)
paths=[paths[i] for i in keep]; y=y[keep]; hashes=hashes[keep]
dup=int(len(hashes)-len(set(hashes.tolist())))
log(f"after dedup: {len(y)} unique images | happy={int((y==1).sum())} sad={int((y==0).sum())} | duplicates={dup}")
assert len(y)>0

# ---- 3. split ----
from sklearn.model_selection import GroupShuffleSplit
def make_split(seed):
    idx=np.arange(len(y))
    g1=GroupShuffleSplit(n_splits=1,test_size=0.30,random_state=seed); tr,tmp=next(g1.split(idx,y,hashes))
    g2=GroupShuffleSplit(n_splits=1,test_size=0.50,random_state=seed); vr,ter=next(g2.split(tmp,y[tmp],hashes[tmp]))
    va,te=tmp[vr],tmp[ter]
    assert len(set(hashes[tr])&set(hashes[te]))==0 and len(set(hashes[tr])&set(hashes[va]))==0
    return tr,va,te

# ---- 4. STREAMING feature extraction (reads from disk in batches; low RAM) ----
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import Dataset,DataLoader
tf=T.Compose([T.Resize(224),T.CenterCrop(224),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
class PathDS(Dataset):
    def __len__(s): return len(paths)
    def __getitem__(s,i):
        im=Image.open(paths[i]).convert("RGB")   # read fresh, released after batch
        return tf(im), i
def build(name):
    if name=="resnet50": m=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc=nn.Identity(); dim=2048
    elif name=="effb0": m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1); m.classifier=nn.Identity(); dim=1280
    elif name=="vit": m=models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1); m.heads=nn.Identity(); dim=768
    return m.eval().to(dev),dim
def extract(name):
    m,dim=build(name); X=np.zeros((len(paths),dim),np.float32)
    dl=DataLoader(PathDS(),batch_size=64,num_workers=2)
    with torch.no_grad():
        for xb,ix in tqdm(dl,desc=name):
            X[ix.numpy()]=m(xb.to(dev)).cpu().numpy()
    del m
    if dev=="cuda": torch.cuda.empty_cache()
    return X

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score,confusion_matrix
from scipy import stats
def metrics(yt,pp): return dict(acc=accuracy_score(yt,pp),f1=f1_score(yt,pp,average="macro"),kappa=cohen_kappa_score(yt,pp))
def summ(rows):
    def ci(x):
        x=np.array(x); m=x.mean(); s=x.std(ddof=1) if len(x)>1 else 0
        h=1.96*s/np.sqrt(len(x)) if len(x)>1 else 0
        return m,s,(m-h,m+h)
    return {k:ci([r[k] for r in rows]) for k in ["acc","f1","kappa"]}
def mcnemar(yt,p1,p2):
    b=np.sum((p1==yt)&(p2!=yt)); c=np.sum((p1!=yt)&(p2==yt))
    return 1.0 if b+c==0 else 1-stats.chi2.cdf((abs(b-c)-1)**2/(b+c),1)

RESULTS={}; STORE={}
log("\n===== FROZEN FEATURES: 3 architectures x 5 seeds =====")
for arch in ["resnet50","effb0","vit"]:
    log(f"extracting {arch} (streaming from disk)...")
    X=extract(arch); rows=[]
    for s in SEEDS:
        tr,va,te=make_split(s)
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced",random_state=s))
        clf.fit(X[tr],y[tr]); pp=clf.predict(X[te]); rows.append(metrics(y[te],pp))
        if s==0: STORE[arch]=(y[te],pp)
    RESULTS[arch]=rows; m=summ(rows)
    log(f"  {arch}: acc={m['acc'][0]:.4f}+/-{m['acc'][1]:.4f}  F1={m['f1'][0]:.4f}  kappa={m['kappa'][0]:.4f}")
    del X

# ---- 5. summary ----
print("\n"+"="*66)
print(" FINAL SUMMARY (mean +/- std over %d seeds; [95%% CI])"%len(SEEDS))
print("="*66)
print(f"{'architecture':16s} {'accuracy':24s} {'macroF1':16s} {'kappa':8s}")
for a in ["resnet50","effb0","vit"]:
    m=summ(RESULTS[a]); ac,f,k=m['acc'],m['f1'],m['kappa']
    print(f"{a:16s} {ac[0]:.3f}+/-{ac[1]:.3f}[{ac[2][0]:.3f},{ac[2][1]:.3f}] {f[0]:.3f}+/-{f[1]:.3f}   {k[0]:.3f}")
print("\n--- McNemar tests (seed 0) ---")
for a,b in [("resnet50","vit"),("resnet50","effb0"),("effb0","vit")]:
    yt,pa=STORE[a]; _,pb=STORE[b]; p=mcnemar(yt,pa,pb)
    print(f"  {a} vs {b}: p={p:.4f} {'(significant)' if p<0.05 else '(n.s.)'}")
for a in ["resnet50","effb0","vit"]:
    yt,pp=STORE[a]; print(f"\nconfusion [{a}] rows=true sad/happy:"); print(confusion_matrix(yt,pp))
print("\nPaste the FINAL SUMMARY + McNemar + confusion blocks back to complete the paper.")
