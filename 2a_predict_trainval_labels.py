import os
import pickle
import numpy as np
import onnxruntime as ort

from tqdm import tqdm
from glob import glob
from collections import defaultdict

from lib.cfg import load_config
from lib.landmarks import load_landmark_model
from lib.predict import get_points


CONFIG = "experiments/029.yaml"
cfg, model_dir = load_config(CONFIG)

# models for prediction
pose_model_path = cfg['export']['t1t2_model_path']
landmark_model_path = cfg['export']['landmark_model_path']

# data
trainval_npz_dir = cfg['data']['npz_path_trainval']
npz_files = glob(f"{trainval_npz_dir}/**/*.npy", recursive=True)
fov = cfg['transforms']['test']['centrecrop'][0]

output_dict = defaultdict(dict)

out_dir = os.path.join("./data", os.path.basename(os.path.dirname(pose_model_path)))  # For PNGs and JSON
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# LOAD MODELS
landmark_model = load_landmark_model(landmark_model_path)
sess = ort.InferenceSession(pose_model_path)

for i, src in tqdm(enumerate(npz_files), total=len(npz_files)):

    # LOAD DATA
    npy = np.load(src)

    (xs_epi, ys_epi), (xs_end, ys_end) = get_points(sess, landmark_model, npy, fov)

    with open(f"{src}_{os.path.basename(pose_model_path)}_points.pickle", 'wb') as f:
        pickle.dump({'xs_epi': xs_epi,
                     'ys_epi': ys_epi,
                     'xs_end': xs_end,
                     'ys_end': ys_end},
                    f)


from moviepy import editor
from moviepy.editor import ImageSequenceClip

