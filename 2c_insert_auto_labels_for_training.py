import os
import pickle
import numpy as np
from rdp import rdp
from glob import glob
from tqdm import tqdm


USE_RDP = True
ACCEPTED_PNGS = "./data/labels/train_val/auto/png"
SOURCE_DIOMS_PATH = "./data/dicoms/train_val"  # Will put the predictions there, as if they were human labelled, with an -AUTO suffix
OUTPUT_LABEL_PATH = "./data/labels/train_val/auto/pickles"
FOV = 256

for png_path in tqdm(glob(os.path.join(ACCEPTED_PNGS, "*.png"))):
    date_dir, seq_dir, pickle_name = os.path.splitext(os.path.basename(png_path))[0].split('__')
    pickle_path = os.path.join(SOURCE_DIOMS_PATH, date_dir, seq_dir, pickle_name)

    if not os.path.exists(pickle_path):
        print(f"Can't find {pickle_path}")
        continue

    output_label_dir = os.path.join(OUTPUT_LABEL_PATH, date_dir, seq_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    target_file_name = pickle_name.split('.npy_')[0]  # T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy_150_0.0009576.pt.onnx_points.pickle -> T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1
    target_file_name += '.npy_AUTO.pickle'
    target_file_path = os.path.join(output_label_dir, target_file_name)

    # if os.path.exists(target_file_path):
    #     continue

    pd = pickle.load(open(pickle_path, 'rb'))
    xs_end, ys_end, xs_epi, ys_epi = pd['xs_end'], pd['ys_end'], pd['xs_epi'], pd['ys_epi']
    points_end = np.array(list(zip(xs_end, ys_end)))  # 1 * n_points * 2
    points_epi = np.array(list(zip(xs_epi, ys_epi)))

    if USE_RDP:
        len_pre_rdp = len(points_end), len(points_epi)
        points_end = rdp(points_end).astype(np.float)
        points_epi = rdp(points_epi).astype(np.float)
        print(f"{len_pre_rdp} -> {len(points_end), len(points_epi)}")

    # Translate into true labelling coords (remember predictions are done on 256 * 256 crops)
    npy_source_dir = os.path.join(SOURCE_DIOMS_PATH, date_dir, seq_dir)
    npy_source_path = os.path.join(npy_source_dir, pickle_name.split('.npy_')[0] + '.npy')
    npy_source_shape = np.load(npy_source_path).shape
    npy_height, npy_width, npy_depth = npy_source_shape

    # Factor in cropping
    top_left = (npy_width - FOV) // 2, (npy_height - FOV) // 2
    points_end = (points_end + top_left)
    points_epi = (points_epi + top_left)

    # Remember height is from bottom for labels
    points_end[:, 1] = npy_height - points_end[:, 1]
    points_epi[:, 1] = npy_height - points_epi[:, 1]

    to_write = {
        'endo': points_end.tolist(),
        'epi': points_epi.tolist()
    }



    with open(target_file_path, 'wb') as f:
        pickle.dump(to_write, f)
