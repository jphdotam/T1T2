import os
import multiprocessing
from collections import defaultdict

from glob import glob
from tqdm import tqdm

from lib.cfg import load_config
from lib.export import export_label

# CONFIG = "./experiments/034_mini.yaml"
CONFIG = "./experiments/036_mini.yaml"
EXCLUDED_FILES_PATH = "./data/blacklist.txt"

# Load config
cfg, model_dir = load_config(CONFIG)
source_path_data_trainval = cfg['export']['dicom_path_trainval']
source_path_label_trainval = cfg['export']['label_path_trainval']
output_training_data_dir = os.path.join(cfg['data']['npz_path_trainval'])
sequences = cfg['export']['source_channels']
label_classes = cfg['export']['label_classes']
gaussian_sigma = cfg['export']['gaussian_sigma']

# Excluded files
with open(EXCLUDED_FILES_PATH) as f:
    excluded_files = f.read().splitlines()


def export_label_helper(paths):
    dicom_path, label_paths = paths
    for label_path in label_paths:
        output_dir = os.path.join(output_training_data_dir,
                                  os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(label_path)))))
        os.makedirs(output_dir, exist_ok=True)
        try:
            export_label(dicom_path, label_path, 'npz', sequences, label_classes, output_dir, gaussian_sigma)
        except Exception as e:
            print(f"Failed {label_path}: {e}")


def blacklisted(label_path, excluded_files):
    study_dir = os.path.basename(os.path.dirname(label_path))
    npy_name = os.path.basename(label_path).split('.npy')[0] + '.npy'
    is_blacklisted = f"{study_dir}/{npy_name}" in excluded_files
    if is_blacklisted:
        print(f"Skipping {label_path} - excluded")
        return True
    else:
        return False


if __name__ == "__main__":

    if not os.path.exists(output_training_data_dir):
        os.makedirs(output_training_data_dir)

    labelpaths_human = glob(os.path.join(source_path_label_trainval, "**", "*npy_HUMAN.pickle"), recursive=True)
    labelpaths_auto = glob(os.path.join(source_path_label_trainval, 'auto', "**", "*npy_AUTO.pickle"), recursive=True)

    dicompaths = glob(os.path.join(source_path_data_trainval, "**/*.npy"), recursive=True)

    print(f"{len(dicompaths)} source files - found {len(labelpaths_human)} human labels and {len(labelpaths_auto)} auto labels")

    labels_by_seq = defaultdict(list)
    for labelpaths in (labelpaths_human, labelpaths_auto):
        for labelpath in labelpaths:
            seq_id = f"{os.path.basename(os.path.dirname(labelpath))}__{os.path.basename(labelpath).split('.npy')[0]}"
            labels_by_seq[seq_id].append(labelpath)

    labelled_dicoms = defaultdict(list)
    for dicom_path in dicompaths:
        seq_id = f"{os.path.basename(os.path.dirname(dicom_path))}__{os.path.basename(dicom_path).split('.npy')[0]}"
        if seq_id in labels_by_seq:
            for label_path in labels_by_seq[seq_id]:
                labelled_dicoms[dicom_path].append(label_path)

    N_WORKERS = multiprocessing.cpu_count() - 4  #// 2

    with multiprocessing.Pool(N_WORKERS) as p:
        for _ in tqdm(p.imap(export_label_helper, labelled_dicoms.items()), total=len(labelled_dicoms)):
            pass
