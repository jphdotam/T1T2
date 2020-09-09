import numpy as np
import skimage.transform
import onnxruntime as ort
import matplotlib.pyplot as plt

from glob import glob
from utils.landmarks import load_landmark_model, perform_cmr_landmark_detection
from utils.tracing import get_epi_end_paths_from_heatmap_and_landmarks

POSE_MODELPATH = "output/models/026/110_0.0010794.pt.onnx"
LANDMARK_MODELPATH = "E:/Data/T1T2_models/CMR_landmark_network_RO_352_E1_352_sax_with_T1_T1T2_LGE_PerfPD_LossMultiSoftProb_KLD_Dice_Pytorch_1.5.1_2020-08-13_20200813_181146.pts"
SRC_FILES = glob("E:/Data/T1T2_peter/**/*.npy", recursive=True)
# SRC_FILES = glob("E:/Data/T1T2_peter/20200415/T1T2_141613_3956841_3956849_201_20200415-165703/*.npy", recursive=True)
FOV = 256


def normalize_data(data, window_centre, window_width):
    window_min = max(0, window_centre - window_width // 2)
    frame = data - window_min
    frame = frame / window_width
    frame = np.clip(frame, 0, 1)
    frame = frame.astype(np.float32)
    return frame


def pad_if_needed(img, min_height, min_width):
    input_height, input_width = img.shape[:2]
    new_shape = list(img.shape)
    new_shape[0] = max(input_height, min_height)
    new_shape[1] = max(input_width, min_width)
    row_from, col_from = 0, 0
    if input_height < min_height:
        row_from = (min_height - input_height) // 2
    if input_width < min_width:
        col_from = (min_width - input_width) // 2
    out = np.zeros(new_shape, dtype=img.dtype)
    out[row_from:row_from+input_height, col_from:col_from+input_width] = img
    return out


def center_crop(img, crop_height, crop_width, centre=None):
    """Either crops by the center of the image, or around a supplied point.
    Does not pad; if the supplied centre is towards the egde of the image, the padded
    area is shifted so crops start at 0 and only go up to the max row/col
    Returns both the new crop, and the top-left coords as a row,col tuple"""
    input_height, input_width = img.shape[:2]
    if centre is None:
        row_from = (input_height - crop_height)//2
        col_from = (input_width - crop_width)//2
    else:
        row_centre, col_centre = centre
        row_from = max(row_centre - (crop_height//2), 0)
        if (row_from + crop_height) > input_height:
            row_from -= (row_from + crop_height - input_height)
        col_from = max(col_centre - (crop_width//2), 0)
        if (col_from + crop_width) > input_width:
            col_from -= (col_from + crop_width - input_width)
    return img[row_from:row_from+crop_height, col_from:col_from+crop_width], (row_from, col_from)


# LOAD MODELS
landmark_model = load_landmark_model(LANDMARK_MODELPATH)
sess = ort.InferenceSession(POSE_MODELPATH)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

for i, src in enumerate(SRC_FILES):
    # LOAD DATA
    npy = np.load(src)
    t1w, t2w, pd, t1, t2 = np.transpose(npy, (2, 0, 1))

    # IDENTIFY POINTS
    t2w_landmark, _top_left_landmark = center_crop(pad_if_needed(t2w, min_height=FOV, min_width=FOV), crop_height=FOV, crop_width=FOV)
    landmark_points, landmark_probs = perform_cmr_landmark_detection(t2w_landmark, model=landmark_model)

    if np.all(landmark_points==-1):
        print(f"Skipping {src} - unable to identify landmarks")
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
    pred_batch = sess.run([output_name], {input_name: img_batch})[0]  # Returns a list of len 1

    (xs_epi, ys_epi), (xs_end, ys_end) = get_epi_end_paths_from_heatmap_and_landmarks(pred_batch[0], landmark_points)

    # preview
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0, hspace=0)

    for i_plot, pred_channel in enumerate((t1_t2_crop[:, :, 1], *pred_batch[0])):
        ax = axes[i_plot]

        # normalize predictions
        pred_channel = pred_channel - pred_channel.min()
        pred_channel = pred_channel / (pred_channel.max() + 1e-8)

        ax.imshow(pred_channel)

        # plot RV insertion point vectors
        for vector in (vector_ant, vector_post):
            xs, ys = vector.transpose((1,0))
            ax.plot(xs, ys)

        ax.plot(xs_epi, ys_epi, 'b-')
        ax.plot(xs_end, ys_end, 'r-')

    fig.suptitle(src)
    fig.show()

    if i == 100:
        break
