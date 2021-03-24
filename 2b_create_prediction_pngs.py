import os
import cv2
import pickle
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from rdp import rdp
from glob import glob
from tqdm import tqdm

from lib.cfg import load_config
from lib.cmaps import default_cmap
from lib.windows import normalize_data
from lib.inference import center_crop, pad_if_needed, get_normalized_channel_stack

CONFIG = "experiments/029.yaml"
cfg, model_dir = load_config(CONFIG)

trainval_npz_dir = cfg['data']['input_path_trainval']
npz_files = glob(f"{trainval_npz_dir}/**/*.npy", recursive=True)
pose_model_path = cfg['data']['t1t2_model_path']
landmark_model_path = cfg['data']['landmark_model_path']
fov = cfg['transforms']['test']['centrecrop'][0]

USE_RDP = True  # Ramer-Douglas-Peucker filtering
DATA_DIR = r"E:/Data/T1T2_HH/npy"
PNG_FOLDER = rf"E:/Data/T1T2_HH/predicted_labels{'_rdp' if USE_RDP else ''}"
FOV = 256

predicted_labels = glob(os.path.join(DATA_DIR, "**/*_points.pickle"), recursive=True)

for predicted_label_path in tqdm(predicted_labels):
    pd = pickle.load(open(predicted_label_path, 'rb'))
    xs_end, ys_end, xs_epi, ys_epi = pd['xs_end'], pd['ys_end'], pd['xs_epi'], pd['ys_epi']
    points_end = np.array([list(zip(xs_end, ys_end))])  # 1 * n_points * 2
    points_epi = np.array([list(zip(xs_epi, ys_epi))])

    if USE_RDP:
        len_pre_rdp = len(points_end[0]), len(points_epi[0])
        points_end = np.expand_dims(rdp(points_end[0]), 0)
        points_epi = np.expand_dims(rdp(points_epi[0]), 0)
        print(f"{len_pre_rdp} -> {len(points_end[0]), len(points_epi[0])}")

    npy_path = predicted_label_path.split('.npy')[0] + '.npy'

    npy = np.load(npy_path)
    t1w, t2w, pd, t1, t2 = np.transpose(npy, (2, 0, 1))

    t1_pre, t1_post, t2, t1w, t2w, pd, t1_t2 = get_normalized_channel_stack(t1, t2, t1w, t2w, pd, data_stack_format='all')
    t1_t2_crop, _top_left = center_crop(pad_if_needed(t1_t2, min_height=FOV, min_width=FOV), crop_height=FOV,
                                        crop_width=FOV)

    img_out = []

    for i_channel in (3, 4, 5):
        img = default_cmap(t1_t2_crop[:, :, i_channel])
        cv2.polylines(img, points_end, True, [1, 1, 1])
        cv2.polylines(img, points_epi, True, [1, 1, 1])
        img_out.append(img)

    img_out = (np.hstack(img_out) * 255).astype(np.uint8)
    #plt.imshow(img_out)

    date = os.path.basename(os.path.dirname(os.path.dirname(predicted_label_path)))
    study = os.path.basename(os.path.dirname(predicted_label_path))
    labelname = os.path.basename(predicted_label_path)

    out_path = os.path.join(PNG_FOLDER, f"{date}__{study}__{labelname}.png")
    skimage.io.imsave(out_path, img_out)

    break
