import os
import multiprocessing

from glob import glob
from tqdm import tqdm

from utils.cfg import load_config
from utils.export import export_label
from utils.windows import SEQUENCE_WINDOWS

CONFIG = "../experiments/026.yaml"
EXCLUDED_FILES_PATH = "data/blacklist.txt"

# Load config
cfg, vis_dir, model_dir = load_config(CONFIG)
npy_dir = cfg['export']['npydir']
output_dir = os.path.join(cfg['data']['pngdir'])
sequences = cfg['export']['sequences']
label_classes = cfg['export']['label_classes']
gaussian_sigma = cfg['export']['gaussian_sigma']
frmt = cfg['export']['format']

# Excluded files
with open(EXCLUDED_FILES_PATH) as f:
    excluded_files = f.read().splitlines()


def export_label_helper(labelpath):
    export_label(labelpath, frmt, sequences, label_classes, output_dir, gaussian_sigma)


def blacklisted(labelpath, excluded_files):
    study_dir = os.path.basename(os.path.dirname(labelpath))
    npy_name = os.path.splitext(os.path.basename(labelpath))[0]
    is_blacklisted = f"{study_dir}/{npy_name}" in excluded_files
    if is_blacklisted:
        print(f"Skipping {labelpath} - excluded")
        return True
    else:
        return False


if __name__ == "__main__":

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    labelpaths = glob(os.path.join(npy_dir, "**", "*npy.pickle"), recursive=True)
    labelpaths = [l for l in labelpaths if not blacklisted(l, excluded_files)]

    N_WORKERS = multiprocessing.cpu_count()

    with multiprocessing.Pool(N_WORKERS) as p:
        for _ in tqdm(p.imap(export_label_helper, labelpaths), total=len(labelpaths)):
            pass
