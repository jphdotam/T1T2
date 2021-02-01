import os
import shutil
from tqdm import tqdm
from glob import glob

SOURCE_ROOT = r"E:\Data\T1T2_all\npy"

source_labels = glob(os.path.join(SOURCE_ROOT, "**/*.npy.pickle"), recursive=True)

for source_label_path in tqdm(source_labels):
    os.rename(source_label_path, source_label_path.replace('.npy.pickle', '.npy_HUMAN.pickle'))
