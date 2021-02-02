import os
import cv2
import json
import numpy as np
import skimage.io
import skimage.transform
import onnxruntime as ort

from glob import glob
from tqdm import tqdm
from collections import defaultdict

import torch

from lib.cfg import load_config
from lib.hrnet import get_hrnet_model, get_hrnet_cfg
from lib.landmarks import load_landmark_model, perform_cmr_landmark_detection
from lib.tracing import get_epi_end_paths_from_heatmap_and_landmarks as get_paths
from lib.inference import center_crop, pad_if_needed
from lib.vis import compute_bullseye_sector_mask_for_slice
from lib.windows import normalize_data


CONFIG = "experiments/030.yaml"
cfg, _ = load_config(CONFIG)

TEST_DICOM_DIR = cfg['export']['dicom_path_test']
POSE_MODELPATH = "./output/models/030/154_0.0004970.pt"
LANDMARK_MODELPATH = cfg['export']['landmark_model_path']
SRC_FILES = glob(os.path.join(TEST_DICOM_DIR, "**/*.npy"), recursive=True)
FOV = 256
WRITE_PNGS = True
DEVICE = "cuda"

model = get_hrnet_model(get_hrnet_cfg(cfg)).to(DEVICE)
model = model.eval()
model.load_state_dict(torch.load(POSE_MODELPATH)['state_dict'])

output_dict = defaultdict(dict)

out_dir = os.path.join("./data", f"predictions_{os.path.basename(os.path.dirname(POSE_MODELPATH))}")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# LOAD MODELS
landmark_model = load_landmark_model(LANDMARK_MODELPATH)

for i, src in tqdm(enumerate(SRC_FILES), total=len(SRC_FILES)):
    # LOAD DATA
    npy = np.load(src)
    t1w, t2w, pd, t1, t2 = np.transpose(npy, (2, 0, 1))
    t1_raw, t2_raw = t1.copy(), t2.copy()

    # IDENTIFY POINTS
    t2w_landmark, _top_left_landmark = center_crop(pad_if_needed(t2w, min_height=FOV, min_width=FOV), crop_height=FOV, crop_width=FOV)
    landmark_points, landmark_probs = perform_cmr_landmark_detection(t2w_landmark, model=landmark_model)

    if np.any(landmark_points==-1):
        print(f"Skipping {src} - unable to identify all landmarks")
        continue

    vector_ant = landmark_points[[0,2]]
    vector_post = landmark_points[[1,2]]

    # POSE MODEL
    t1_pre = normalize_data(t1, window_centre=1300.0, window_width=1300.0)
    t1_post = normalize_data(t1, window_centre=500.0, window_width=1000.0)
    t2 = normalize_data(t2, window_centre=60.0, window_width=120.0)
    t1w = t1w - t1w.min()
    t1w /= t1w.max()
    t2w = t2w - t2w.min()
    t2w /= t2w.max()
    pd = pd - pd.min()
    pd /= pd.max()
    t1_pre = (t1_pre*255).astype(np.uint8)
    t1_post = (t1_post*255).astype(np.uint8)
    t2 = (t2*255).astype(np.uint8)
    t1w = (t1w*255).astype(np.uint8)
    t2w = (t2w*255).astype(np.uint8)
    pd = (pd*255).astype(np.uint8)

    t1_t2 = np.dstack((t1w, t2w, pd, t1_pre, t1_post, t2))
    t1_t2_crop, _top_left = center_crop(pad_if_needed(t1_t2, min_height=FOV, min_width=FOV), crop_height=FOV, crop_width=FOV)
    t1_t2_double = skimage.transform.rescale(t1_t2_crop, 2, order=3, multichannel=True)

    t1_t2_in = t1_t2_double.transpose((2, 0, 1))

    img_batch = np.expand_dims(t1_t2_in, 0).astype(np.float32)
    x = torch.from_numpy(img_batch).float().to(DEVICE)

    with torch.no_grad():
        pred_batch = model(x).cpu().numpy()

    # rv masks
    rvi1_xy, rvi2_xy, lv_xy = landmark_points
    rvimid_xy = 0.5 * (rvi1_xy + rvi2_xy)
    rv_xy = lv_xy + 2 * (rvimid_xy - lv_xy)
    mask_rvi1 = np.zeros_like(t2w)
    mask_rvi1[int(round(rvi1_xy[1])), int(round(rvi1_xy[0]))] = 1
    mask_rvmid = np.zeros_like(t2w)
    mask_rvmid[int(round(rv_xy[1])), int(round(rv_xy[0]))] = 1

    # Lv ridge tracing using landmarks
    if np.all(landmark_points == -1):
        print(f"Was unable to find landmarks on sample {i}")
        (xs_epi, ys_epi), (xs_end, ys_end) = [[], []]
    else:
        (xs_epi, ys_epi), (xs_end, ys_end) = get_paths(pred_batch[0], landmark_points)

    # ridges to masks
    mask_lvcav, mask_lvwall = np.zeros_like(t2w_landmark, dtype=np.uint8), np.zeros_like(t2w_landmark, dtype=np.uint8)
    points_end = np.array([list(zip(xs_end, ys_end))])
    points_epi = np.array([list(zip(xs_epi, ys_epi))])
    color = np.uint8(np.ones(3) * 1).tolist()
    cv2.fillPoly(mask_lvcav, points_end, color)
    cv2.fillPoly(mask_lvwall, points_epi, color)

    # sectors
    sectors, sectors_32 = compute_bullseye_sector_mask_for_slice(mask_lvcav, mask_lvwall, mask_rvmid, mask_rvi1, 6)

    # window maps
    t1, _ = center_crop(pad_if_needed(t1_raw, min_height=FOV, min_width=FOV), crop_height=FOV, crop_width=FOV)
    t2, _ = center_crop(pad_if_needed(t2_raw, min_height=FOV, min_width=FOV), crop_height=FOV, crop_width=FOV)

    # for i_plot, pred_channel in enumerate((t1_t2_crop[:, :, 1], *pred_batch[0])):
    #     ax = axes[i_plot]
    #
    #     # normalize predictions
    #     pred_channel = pred_channel - pred_channel.min()
    #     pred_channel = pred_channel / (pred_channel.max() + 1e-8)
    #
    #     ax.imshow(pred_channel)
    #
    #     # plot RV insertion point vectors
    #     for vector in (vector_ant, vector_post):
    #         xs, ys = vector.transpose((1,0))
    #         ax.plot(xs, ys)
    #
    #     ax.plot(xs_epi, ys_epi, 'b-')
    #     ax.plot(xs_end, ys_end, 'r-')
    #
    # fig.suptitle(src)
    # fig.show()

    for seq_name, seq_img in zip(('t1', 't2'), (t1, t2)):
        for segment_id in list(range(1, 6+1)):
            seg_values = seq_img[sectors == segment_id]
            study_id = f"{os.path.basename(os.path.dirname(src))}__{os.path.basename(src)}"
            output_dict[study_id][f"{seq_name}_{segment_id}"] = {
                'count': len(seg_values),
                'mean': np.nanmean(seg_values),
                'median': np.nanmedian(seg_values)
            }
            if WRITE_PNGS:
                png_path = os.path.join(out_dir, study_id + ".png")
                skimage.io.imsave(png_path, sectors.astype(np.uint8), check_contrast=False)

with open(os.path.join(out_dir, "segments.json"), 'w') as f:
    json.dump(output_dict, f, indent=4)
