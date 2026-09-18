import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12,3))
ax.set_xlim(0,12); ax.set_ylim(0,2); ax.axis("off")
boxes = ["data/raw\n392MB","make_dataset\ntemporal","build_features\n12+8","LGBM\nisotonic\nBrier 0.155","profit\nthr 0.05","FastAPI\n/predict\n/explain","MLflow\n+ tests"]
x=0.2
for b in boxes:
    ax.add_patch(patches.Rectangle((x,0.7),1.4,0.8, fill=False))
    ax.text(x+0.7,1.1,b, ha="center", va="center", fontsize=7)
    if x>0.2:
        ax.arrow(x-0.05,1.1,0.15,0, head_width=0.08, head_length=0.08, fc="k")
    x+=1.65
plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=150)
print("saved docs/architecture.png")