import os
import pickle
from glob import glob

OLD_PETER_DIR = "E:/Data/T1T2_peter_old"  # _old will be removed

def has_new_file(picklepath):
    orig_npy = os.path.splitext(picklepath)[0]
    new_npy = orig_npy.replace('_old', '')
    valid = os.path.exists(new_npy)
    if valid:
        return True
    else:
        print(f"Could not find file {new_npy}")
        return False

old_peter_labels = glob(os.path.join(OLD_PETER_DIR, "**/*.pickle"), recursive=True)
matched_labels = [f for f in old_peter_labels if has_new_file(f)]

print(f"{len(matched_labels)} of {len(old_peter_labels)} have a matching source file")