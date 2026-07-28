# ============================================================================
#  KIDO — make a SAMPLE DRAWINGS figure (real images) for the paper
#  Shows real happy vs sad children's drawings. Optionally, if you have a
#  trained model's predictions, it can show correct vs misclassified ones.
#  Run AFTER the data is extracted (reuses /content/kido_extracted).
#  Produces sample_drawings.png -- download it and send it to me.
# ============================================================================
import os, glob, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

EXDIR="/content/kido_extracted"
jpgs=glob.glob(f"{EXDIR}/**/*.jpg",recursive=True)+glob.glob(f"{EXDIR}/**/*.jpeg",recursive=True)+glob.glob(f"{EXDIR}/**/*.png",recursive=True)

# split by emotion from filename (...-H.jpg / ...-S.jpg)
happy=[p for p in jpgs if os.path.splitext(os.path.basename(p))[0].split("-")[-1].upper()=="H"]
sad  =[p for p in jpgs if os.path.splitext(os.path.basename(p))[0].split("-")[-1].upper()=="S"]
random.seed(7)
happy_s=random.sample(happy,4)
sad_s  =random.sample(sad,4)

fig,axes=plt.subplots(2,4,figsize=(11,5.6))
for ax,p in zip(axes[0],happy_s):
    ax.imshow(Image.open(p).convert("RGB")); ax.axis("off")
for ax,p in zip(axes[1],sad_s):
    ax.imshow(Image.open(p).convert("RGB")); ax.axis("off")
axes[0,0].set_ylabel("Happiness",fontsize=13,fontweight="bold",rotation=90,labelpad=15)
axes[1,0].set_ylabel("Sadness",fontsize=13,fontweight="bold",rotation=90,labelpad=15)
# re-enable the y-label axis (imshow turned axis off)
for row,lab in [(0,"Happiness"),(1,"Sadness")]:
    axes[row,0].axis("on"); axes[row,0].set_xticks([]); axes[row,0].set_yticks([])
    axes[row,0].set_ylabel(lab,fontsize=13,fontweight="bold")
    for s in axes[row,0].spines.values(): s.set_visible(False)
plt.suptitle("Example KIDO drawings by self-reported emotion",fontsize=13,fontweight="bold",y=0.98)
plt.tight_layout()
plt.savefig("sample_drawings.png",dpi=200,bbox_inches="tight")
print("saved sample_drawings.png -- download it and send it to me to place in the paper")
