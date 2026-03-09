import os
import shutil
import random
from pathlib import Path
import pandas as pd

random.seed(42)

src_dir = "/projects/memaro/rpujol/unfolding/data/dataset1"
dst_dir = "/projects/memaro/mcodjo/pipeline/data/dataset1"

# Indices a deplacer du train vers val
all_train = list(range(67))
extra     = sorted(random.sample(all_train, 10))
new_train = sorted([i for i in all_train if i not in extra])

print(f"Train original  : 67 instances")
print(f"Extra -> val    : {extra}")
print(f"Nouveau train   : {len(new_train)} instances")
print(f"Nouveau val     : {14 + len(extra)} instances")
print(f"Test            : 15 instances (inchange)")

sous_dossiers = ["ground_truth", "input", "masques"]

# ============================================================
# TRAIN : new_train indices renumerotes 0..N
# ============================================================
split = "train"
for sub in sous_dossiers:
    Path(f"{dst_dir}/{split}/{sub}").mkdir(parents=True, exist_ok=True)

for new_idx, old_idx in enumerate(new_train):
    for sub in sous_dossiers:
        shutil.copy2(
            f"{src_dir}/{split}/{sub}/{old_idx}.npy",
            f"{dst_dir}/{split}/{sub}/{new_idx}.npy"
        )

train_params = pd.read_csv(f"{src_dir}/train/params.csv")
new_train_params = train_params.iloc[new_train].reset_index(drop=True)
new_train_params.to_csv(f"{dst_dir}/train/params.csv", index=False)
print(f"\n[OK] train : {len(new_train)} fichiers")

# ============================================================
# VAL 
# ============================================================
split = "val"
for sub in sous_dossiers:
    Path(f"{dst_dir}/{split}/{sub}").mkdir(parents=True, exist_ok=True)

# Val original
for new_idx in range(14):
    for sub in sous_dossiers:
        shutil.copy2(
            f"{src_dir}/val/{sub}/{new_idx}.npy",
            f"{dst_dir}/val/{sub}/{new_idx}.npy"
        )

# Extra depuis train
for i, old_idx in enumerate(extra):
    for sub in sous_dossiers:
        shutil.copy2(
            f"{src_dir}/train/{sub}/{old_idx}.npy",
            f"{dst_dir}/val/{sub}/{14 + i}.npy"
        )

val_params   = pd.read_csv(f"{src_dir}/val/params.csv")
extra_params = train_params.iloc[extra].reset_index(drop=True)
new_val_params = pd.concat([val_params, extra_params], ignore_index=True)
new_val_params.to_csv(f"{dst_dir}/val/params.csv", index=False)
print(f"[OK] val   : {14 + len(extra)} fichiers")

# ============================================================
# TEST 
# ============================================================
split = "test"
for sub in sous_dossiers:
    Path(f"{dst_dir}/{split}/{sub}").mkdir(parents=True, exist_ok=True)

for idx in range(15):
    for sub in sous_dossiers:
        shutil.copy2(
            f"{src_dir}/test/{sub}/{idx}.npy",
            f"{dst_dir}/test/{sub}/{idx}.npy"
        )

shutil.copy2(f"{src_dir}/test/params.csv", f"{dst_dir}/test/params.csv")
print(f"[OK] test  : 15 fichiers")

# ============================================================
# RESUME
# ============================================================
print(f"\n[DONE] Dataset2 cree dans {dst_dir}")
print(f"       train({len(new_train)}) / val({14 + len(extra)}) / test(15)")
print(f"\nMets a jour ta config :")
print(f'       "data_dir": "{dst_dir}"')
print(f'       "train_instances": {len(new_train)}')
print(f'       "val_instances"  : {14 + len(extra)}')
print(f'       "test_instances" : 15')