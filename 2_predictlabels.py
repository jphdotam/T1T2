import os
import pickle
import numpy as np
import onnxruntime as ort

from tqdm import tqdm
from glob import glob
from collections import defaultdict

from utils.landmarks import load_landmark_model
from utils.predict import get_points

POSE_MODELPATH = "output/models/028/150_0.0009576.pt.onnx"
LANDMARK_MODELPATH = "E:/Data/T1T2_models/CMR_landmark_network_RO_352_E1_352_sax_with_T1_T1T2_LGE_PerfPD_LossMultiSoftProb_KLD_Dice_Pytorch_1.5.1_2020-08-13_20200813_181146.pts"
SRC_FILES = glob("E:/Data/T1T2_HH/npy/**/*.npy", recursive=True)
FOV = 256

output_dict = defaultdict(dict)

out_dir = os.path.join("./data", os.path.basename(os.path.dirname(POSE_MODELPATH)))  # For PNGs and JSON
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# LOAD MODELS
landmark_model = load_landmark_model(LANDMARK_MODELPATH)
sess = ort.InferenceSession(POSE_MODELPATH)

for i, src in tqdm(enumerate(SRC_FILES), total=len(SRC_FILES)):

    # LOAD DATA
    npy = np.load(src)

    (xs_epi, ys_epi), (xs_end, ys_end) = get_points(sess, landmark_model, npy, FOV)

    with open(f"{src}_{os.path.basename(POSE_MODELPATH)}_points.pickle", 'wb') as f:
        pickle.dump({'xs_epi': xs_epi,
                     'ys_epi': ys_epi,
                     'xs_end': xs_end,
                     'ys_end': ys_end},
                    f)
